# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 true \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-SFT
"""
from dotenv import load_dotenv
load_dotenv()

import logging
import os
import re
import ast
import json
import sys

import datasets
import transformers
from typing import List, Dict, Any

from datasets import Dataset, DatasetDict
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from alignment import ScriptArguments, SFTConfig, get_dataset, get_model, get_tokenizer
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format

logger = logging.getLogger(__name__)



def compose_messages_with_thinking(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Recompose parsed messages into the serialized conversation format:
    
    [
      {"role": "user", "content": "~~"},
      {"role": "assistant", "content": "<think>...</think>\n<tool_call>...</tool_call>"},
      {"role": "tool", "content": "~~"},
      {"role": "assistant", "content": "~~"},
      ...
    ]

    Rules:
      - Assistant messages may contain:
          * reasoning_content  → wrapped in <think>...</think>
          * tool_calls          → each call wrapped separately in <tool_call>...</tool_call>
          * content            → plain text (public output)
      - Consecutive assistant messages (reasoning/tool/content) are merged into one.
      - Tool and user messages remain as-is.
    """
    composed: List[Dict[str, Any]] = []
    buffer: Dict[str, Any] = None  # holds partial assistant message

    def flush_buffer():
        """If an assistant message is being built, finalize and append it."""
        nonlocal buffer
        if buffer is not None:
            content_parts = []

            # Handle reasoning section
            if "reasoning_content" in buffer and buffer["reasoning_content"]:
                content_parts.append(f"<think>\n{buffer['reasoning_content'].strip()}\n</think>")
            else:
                content_parts.append(f"<think>\n\n</think>")

            # Handle tool calls
            if "tool_calls" in buffer and buffer["tool_calls"]:
                tool_calls = buffer["tool_calls"]

                # If it's a list, serialize each individually
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        tool_str = json.dumps(tool_call, ensure_ascii=False)
                        content_parts.append(f"<tool_call>\n{tool_str}\n</tool_call>")
                # Single tool_call (dict)
                elif isinstance(tool_calls, dict):
                    tool_str = json.dumps(tool_calls, ensure_ascii=False)
                    content_parts.append(f"<tool_call>\n{tool_str}\n</tool_call>")
                # Fallback: raw string
                else:
                    content_parts.append(f"<tool_call>\n{str(tool_calls)}\n</tool_call>")

            # Handle normal assistant text
            if "content" in buffer and buffer["content"]:
                content_parts.append(buffer["content"].strip())

            # Combine all parts
            final_content = "\n".join(content_parts).strip()
            composed.append({"role": "assistant", "content": final_content})
            buffer = None

    # Iterate through messages
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            # Start or continue assistant message
            if buffer is None:
                buffer = {"role": "assistant"}

            # Merge fields accordingly
            if "reasoning_content" in msg:
                buffer["reasoning_content"] = (
                    buffer.get("reasoning_content", "") + "\n" + msg["reasoning_content"]
                ).strip()
            elif "tool_calls" in msg:
                buffer["tool_calls"] = msg["tool_calls"]
            elif "content" in msg:
                buffer["content"] = (
                    buffer.get("content", "") + "\n" + msg["content"]
                ).strip()
        else:
            flush_buffer()
            composed.append(msg)

    # Flush any remaining assistant block
    flush_buffer()
    return composed

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################
    def parse_tools_if_needed(tools_field):
        """
        Check if tools field needs JSON parsing and parse if necessary.
        
        Args:
            tools_field: The tools field from the dataset example
            
        Returns:
            Parsed tools as Python object (list of dicts) or raise error if parsing fails
        """
        if not tools_field:
            raise ValueError("tools_field is empty")
        
        if isinstance(tools_field, list):
            return tools_field
        
        if isinstance(tools_field, str):
            parsed_tools = json.loads(tools_field)
            return parsed_tools
    
    def tokenize_messages(example: dict, tokenizer: Any):
        messages = json.loads(example["messages"])
        messages = compose_messages_with_thinking(messages)
        processed = tokenizer.apply_chat_template(
            messages,
            return_dict=True,
            return_assistant_tokens_mask=True,
            tools=parse_tools_if_needed(example["tools"]),
            **example.get("chat_template_kwargs", {}),
        )
        if "assistant_masks" in processed and 1 not in processed["assistant_masks"]:
            raise RuntimeError(
                "You're using `assistant_only_loss=True`, but at least one example has no "
                "assistant tokens. This usually means the tokenizer's chat template doesn't "
                "generate assistant masks — it may be missing the `{% generation %}` keyword. Please "
                "check the template and ensure it's correctly configured to support assistant "
                "masking."
            )
        output = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}
        return output
    
    dataset = get_dataset(script_args)
    
    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    dataset = dataset.map(
        tokenize_messages,
        num_proc=training_args.dataset_num_proc,
        load_from_cache_file=False,
        fn_kwargs={"tokenizer": tokenizer}
    )
    dataset = dataset.filter(
        lambda x: len(x["input_ids"]) < training_args.max_length, load_from_cache_file=False
    )
    
    ############
    # Load model
    ############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    if tokenizer.chat_template is None:
        logger.info("No chat template provided, using ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.model.config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "model_name": training_args.hub_model_id if training_args.push_to_hub else None,
        "dataset_name": script_args.dataset_name,
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
