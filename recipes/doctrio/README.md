# Instructions to train doctrio-14B

## Setup

Make sure you followed the installation instructions in the [README.md](README.md) file. We tested the training setup with 8 GPUs (80GB of VRAM) to train the full model.

## Full training examples

```shell
# - SFT
# Qwen3-14B
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml recipes/doctrio/run_sft.py --config recipes/doctrio/sft/qwen-sft.yaml

# Exaone-3.5-7.8B
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml recipes/doctrio/run_sft.py --config recipes/doctrio/sft/exaone-sft.yaml
```