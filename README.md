# distributed-experiments [Custom DDP and FSDP Implementation for nanoGPT]

This repository contains custom implementations of PyTorch's Distributed Data Parallel (DDP) and Fully Sharded Data Parallel (FSDP) training strategies, specifically tailored for [nanoGPT](https://github.com/karpathy/nanoGPT). The implementations focus on providing efficient distributed training capabilities while maintaining simplicity and readability.

## CustomDDP Features

- **Automatic Parameter Bucketing**: Adjustable bucket sizes to tune communication overhead
- **Asynchronous Communication**: Overlaps computation and communication during backward pass
- **Configurable Gradient Synchronization**: Interface for manual control over synchronization timing

## CustomFSDP Features

- **Deferred Initialization**: Model parameters are initialized progressively FSDP unit by FSDP unit, with each unit being immediately sharded after initialization, rather than initializing the entire model at once
- **Memory Efficient**: Eliminates temporary memory allocations during all-gather and reduce-scatter by transferring ownership between `flat_param` and `local_shard` data pointers, rather than creating intermediate buffers
- **Parameter Sharing**: Maintains correct parameter sharing semantics across the model

## Usage

For distributed training, use `torchrun` to launch the training script across multiple GPUs:

```bash
torchrun --standalone --nproc_per_node=N train.py [args]
```

where N is the number of GPUs.

### Training Arguments

- `--data_parallel_type`: Choose parallelization strategy
  - `ddp`: Distributed Data Parallel
  - `fsdp`: Fully Sharded Data Parallel

- `--implementation`: Select implementation type
  - `pytorch`: Use PyTorch's native implementation
  - `custom`: Use custom implementation

- `--gradient_clipping`: Enable gradient clipping

### Example Commands

```bash
# Run with 4 GPUs using custom DDP implementation
torchrun --standalone --nproc_per_node=4 train.py --data_parallel_type ddp --implementation custom

# Run with 8 GPUs using custom FSDP implementation with gradient clipping
torchrun --standalone --nproc_per_node=8 train.py --data_parallel_type fsdp --implementation custom --gradient_clipping

# Single GPU training with PyTorch DDP
python train.py --data_parallel_type ddp --implementation pytorch
```