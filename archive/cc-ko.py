import os

os.environ["HF_TOKEN"] = "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"
os.environ["HF_HOME"] = "/gpfs/home/exaone/.cache/huggingface/"

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
from datatrove.pipeline.extractors import Trafilatura

from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter
)
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages

from alignment.pipeline.filters.korean_filter import KoreanQualityFilter

DUMP_TO_PROCESS = "CC-MAIN-2025-26"

"""
    Base processing pipeline
"""

MAIN_OUTPUT_PATH = f"/gpfs/home/eungizoa/data/kfmai/cc/{DUMP_TO_PROCESS}/main"
FILTERING_OUTPUT_PATH = f"/gpfs/home/eungizoa/data/kfmai/cc/{DUMP_TO_PROCESS}/filter"

if __name__ == "__main__":

    base_processing_executor = LocalPipelineExecutor(
        pipeline=[
            WarcReader(
                f"s3://commoncrawl/crawl-data/{DUMP_TO_PROCESS}/segments/",
                glob_pattern="*/warc/*",
                default_metadata={"dump": DUMP_TO_PROCESS},
            ),
            URLFilter(exclusion_writer=None),
            Trafilatura(favour_precision=True, timeout=10),
            LanguageFilter(
                languages=["kor_Hang"],
                language_threshold=0.7,
                backend="glotlid",
                label_only=False,
                exclusion_writer=None,
                # exclusion_writer=JsonlWriter(
                #     f"{FILTERING_OUTPUT_PATH}/base_processing/language",
                #     output_filename="${language}/" + "/${rank}.jsonl.gz", # folder structure: language/file
                # ),
            ),
            GopherRepetitionFilter(
                exclusion_writer=None,
                # exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/base_processing/gopher_rep")
            ),
            # Replace GoPherQualityFilter with self-made KoreanQualityFilter
            KoreanQualityFilter(
                exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/base_processing/korean_qual")
            ),
            C4QualityFilter(
                filter_no_terminal_punct=False,
                exclusion_writer=None
                # exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/base_processing/c4_qual"),
            ),
            FineWebQualityFilter(
                line_punct_thr=0.11,      
                exclusion_writer=None 
                # exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/base_processing/fineweb_qual")
            ),
            JsonlWriter(f"{FILTERING_OUTPUT_PATH}/base_processing")
        ],
        tasks=1024, # considering number of files; If the number of files is 1, then set "task" as 1 and "workers" as 1 cause only one process than others would receive file shard (a whole file in this case). 
        workers=128,
        logging_dir=f"{MAIN_OUTPUT_PATH}/logs/base_processing",
        randomize_start_duration=60,
        local_tasks=-1,
        local_rank_offset=0,
    )

    base_processing_executor.run()

    """
        MinHash deduplication to each individual dump
    """
    minhash_config = MinhashConfig(
        hash_config=HashConfig(
            precision=64,
            hash_fc="xxhash"
        ),
        num_buckets=14,
        hashes_per_bucket=8,
        n_grams=5
    )

    MINHASH_BASE_PATH = f"{FILTERING_OUTPUT_PATH}/minhash"

    # stage 1 computes minhash signatures for each task
    # signatures/
    # ├── bucket_000/
    # │   ├── 00000.minhash.sig  # rank0 signatures
    # │   ├── 00001.minhash.sig  # rank1 signatures
    # │   └── ...
    # ├── bucket_001/
    # │   ├── 00000.minhash.sig
    # │   └── ...
    # └── ...
    stage1 = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                f"{FILTERING_OUTPUT_PATH}/base_processing",
                recursive=False,
                file_progress=True,
                doc_progress=True
            ),
            MinhashDedupSignature(
                output_folder=f"{MINHASH_BASE_PATH}/signatures",
                config=minhash_config,
                language=Languages.korean__hang
            ),
        ],
        tasks=1024,
        workers=128,
        logging_dir=f"{MAIN_OUTPUT_PATH}/logs/minhash/signatures",
        randomize_start_duration=60,
        local_tasks=-1,
        local_rank_offset=0,
        depends=base_processing_executor,
    )

    # stage 2 find duplicates in each bucket file
    # buckets/
    # ├── 00000_00.dups  # bucket 0, rank0 duplicates
    # ├── 00001_00.dups  # bucket 1, rank0 duplicates
    # └── ...
    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{MINHASH_BASE_PATH}/signatures",
                output_folder=f"{MINHASH_BASE_PATH}/buckets",
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets * 80, # the code supports parallelizing each bucket.
        workers=128, 
        logging_dir=f"{MAIN_OUTPUT_PATH}/logs/minhash/buckets",
        randomize_start_duration=60,
        local_tasks=-1,
        local_rank_offset=0,
        depends=stage1,
    )

    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{MINHASH_BASE_PATH}/buckets",
                output_folder=f"{MINHASH_BASE_PATH}/remove_ids",
                config=minhash_config,
            ),
        ],
        tasks=1, # this step runs on a single task, World size must be 1 for clustering
        workers=128,
        logging_dir=f"{MAIN_OUTPUT_PATH}/logs/minhash/clusters",
        randomize_start_duration=60,
        local_tasks=-1,
        local_rank_offset=0,
        depends=stage2,
    )

    final_stage = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                f"{FILTERING_OUTPUT_PATH}/base_processing",
                recursive=False,
                file_progress=True,
                doc_progress=True
            ),
            TokensCounter(tokenizer_name_or_path="Qwen/Qwen3-1.7B"),
            MinhashDedupFilter(input_folder=f"{MINHASH_BASE_PATH}/remove_ids"),
            PIIFormatter(),
            JsonlWriter(f"{MAIN_OUTPUT_PATH}"),
        ],
        tasks=1024,
        workers=128,
        logging_dir=f"{MAIN_OUTPUT_PATH}/logs/final_stage",
        randomize_start_duration=60,
        local_tasks=-1,
        local_rank_offset=0,
        depends=stage3,
    )

    final_stage.run()