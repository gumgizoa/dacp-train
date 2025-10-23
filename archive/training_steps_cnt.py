from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader, TensorDataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def create_mock_dataloader(num_samples=1000, batch_size=8):
    """Create a mock dataloader with specified number of samples"""
    # Create dummy data
    data = torch.randn(num_samples, 10)  # 1000 samples, 10 features each
    labels = torch.randint(0, 2, (num_samples,))  # Binary labels
    
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def calculate_training_steps_before_prepare(dataloader, gradient_accumulation_steps, num_epochs=1):
    """Calculate training steps before accelerator.prepare()"""
    steps_per_epoch = len(dataloader) // gradient_accumulation_steps
    total_steps = steps_per_epoch * num_epochs
    return total_steps, steps_per_epoch

def calculate_training_steps_after_prepare(prepared_dataloader, gradient_accumulation_steps, num_epochs=1):
    """Calculate training steps after accelerator.prepare()"""
    steps_per_epoch = len(prepared_dataloader) // gradient_accumulation_steps
    total_steps = steps_per_epoch * num_epochs
    return total_steps, steps_per_epoch

def main():
    
    # Configuration
    num_samples = 1000
    batch_size = 8
    gradient_accumulation_steps = 2
    num_epochs = 3
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16"  # Optional: use mixed precision
    )
    
    num_gpus = accelerator.num_processes
    
    # Only print from main process to avoid duplicate output
    if accelerator.is_main_process:
        print("=== Training Steps Calculation Comparison ===")
        print(f"Configuration:")
        print(f"  - Total samples: {num_samples}")
        print(f"  - Batch size per device: {batch_size}")
        print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  - Number of GPUs: {num_gpus}")
        print(f"  - Number of epochs: {num_epochs}")
        print()
    
    # Create mock dataloader
    dataloader = create_mock_dataloader(num_samples, batch_size)
    
    # Prepare dataloader with accelerator
    prepared_dataloader = accelerator.prepare(dataloader)
    
    if accelerator.is_main_process:
        print("=== AFTER PREPARE ===")
        print(f"Prepared dataloader length: {len(prepared_dataloader)} batches")
        print(f"Effective batch size per device: {batch_size * gradient_accumulation_steps}")
        print(f"Total effective batch size (all devices): {batch_size * gradient_accumulation_steps * accelerator.num_processes}")
    
    # Calculate steps after prepare
    total_steps_after, steps_per_epoch_after = calculate_training_steps_after_prepare(
        prepared_dataloader, gradient_accumulation_steps, num_epochs
    )
    
    if accelerator.is_main_process:
        print(f"Steps per epoch (after prepare): {steps_per_epoch_after}")
        print(f"Total training steps (after prepare): {total_steps_after}")
        print()
        
    if accelerator.is_main_process:
        print("""------Resume training simulation------""")
    optimizer_updated_steps = 60
    completed_steps = optimizer_updated_steps * gradient_accumulation_steps
    starting_epoch = completed_steps // len(prepared_dataloader)
    resume_steps = completed_steps - starting_epoch * len(prepared_dataloader)
    if accelerator.is_main_process:
        print(f"""Resume training from Optimizer updated steps: {optimizer_updated_steps}
Real completed steps during training: {completed_steps}
Starting epoch: {starting_epoch}
Steps to skip at the starting epoch: {resume_steps}""")
    
    for epoch in range(starting_epoch, num_epochs):
        
        if epoch==starting_epoch and resume_steps:
            active_dataloader = accelerator.skip_first_batches(prepared_dataloader, resume_steps)
            if accelerator.is_main_process:
                print(f"Num samples in dataloader: {len(prepared_dataloader.dataset)} vs. active_dataloader: {len(active_dataloader.dataset)}")
        else:
            active_dataloader = prepared_dataloader
    
    accelerator.end_training()

if __name__ == "__main__":
    main()


