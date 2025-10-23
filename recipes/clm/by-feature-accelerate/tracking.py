#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

from dotenv import load_dotenv
load_dotenv()

import argparse
import os
import evaluate
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
# Showcase the experiment tracking capability
from accelerate import Accelerator, DistributedType

MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

def get_dataloaders(accelerator: Accelerator, batch_size: int=16):
    """Creates a set of DataLoader for the `glue` dataset
    
    Args:
        accelerator (`Accelerator`):
            An Accelerator object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = load_dataset("glue", "mrpc")
    
    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs
    
    # Apply the methods `tokenize_function` to all the examples in all the splits of the dataset
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"] # org columns: sentence1, sentence2, label, idx
        )
    # Rename the `label` column to `labels` which is the expected name for labels by the models of `transformers`.
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    def collate_fn(examples):
        # For Torchxla, it's best to pad everything to the same length or training will be very slow.
        max_length = 128 if accelerator.distributed_type == DistributedType.XLA else None
        # When using mixed precision we round multiples of 8/16:
        # Tensor Cores in GPUs (like NVIDIA's Ampere & Hopper) process data in fixed size.        
        # 1. FP8 needs multiples of 16, FP16/BF16 need mutiples of 8 for tensor core optimization; FP32 doesn't require padding.
        # 2. Padding ensures efficient memory alignment & avoids wasted memory.
        # 3. Faster computation by optimizing GPU kernel execution.
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no": # fp16, bf16, etc. ; no means fp32
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None
        
        return tokenizer.pad(
            examples,
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt"
        )
    
    # Instantiate dataloaders.
    # set to ``True`` to drop the last incomplete batch,
    # if the dataset size is not divisible by the batch size. If ``False`` and
    # the size of dataset is not divisible by the batch size, then the last batch will be smaller.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size, drop_last=True
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=EVAL_BATCH_SIZE,
        drop_last=(accelerator.mixed_precision == "fp8"),
    )
    
    return train_dataloader, eval_dataloader

def training_function(config, args):
    # Initialize accelerator
    if args.with_tracking:
        # We pass in "all" to `log_with` to grab all available trackers in the environment
        # Note: If using a custom `Tracker` class, should be passed in here such as:
        # >>> log_with = ["all", MyCustomTrackerClassInstance()]
        accelerator = Accelerator(
            cpu=args.cpu, mixed_precision=args.mixed_precision, log_with="wandb", project_dir=args.project_dir
        )
    else:    
        accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)
    
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    
    metric = evaluate.load("glue", "mrpc")
    
    # If the batch size is too big, we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.XLA:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE
    
    set_seed(seed)
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
    
    # Instantiate the model (we build the model here so that the seed also control new weight initialization)
    # return_dict = True => ModelOutput (Ordered Dictionary)
    # return_dict = False => Tuple
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True, num_labels=2)
    
    # `device_placement` -> controls library automatically moving models, optimizers and data to the correct device
    # We could avoid this line since the accelerator is set with `device_placement=True` (default value True)
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to makeus aware of that)
    model = model.to(accelerator.device)
    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)
    
    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
    )
    
    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects 
    # in the same order we gave them to the `prepare()` method.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # Initialize the trackers we use. Overall configurations can also be stored
    if args.with_tracking:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, config)
    
    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        # Log the total loss of each epoch
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`
            batch.to(accelerator.device)
            outputs = model(**batch)
            loss = outputs.loss
            
            if args.with_tracking:
                total_loss += loss.detach().float()
            
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                if args.with_tracking:
                    accelerator.log(
                        {"train/loss": loss.detach().float()}
                    )
                    
        if args.with_tracking:
            accelerator.log(
                {"train/epoch_loss": total_loss.item() / len(train_dataloader)},
            )
                
        
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references
            )
        
        eval_metric = metric.compute()
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", eval_metric)
        
        # To acutally log, we call `Accelerator.log`
        # The values passed can be of [str, int, float, dict[str, float/int]]
        if args.with_tracking:
            accelerator.log(
                {
                    "eval/accuracy": eval_metric["accuracy"],
                    "eval/fi": eval_metric["f1"],
                    "eval/epoch": epoch
                }
            )
        
    # Runs any special end training behaviors, such as stopping trackers on the main process only.
    # Should always be called at the end of your script if using experiment tracking.
    accelerator.end_training()

def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU."
        "Note. It would override the `mixed_precision` specified in accelerator configuration yaml."
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs` and relevent project information",
    )
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
    training_function(config, args)
    
if __name__ == "__main__":
    main()