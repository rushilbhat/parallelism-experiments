# Parallelism Experiments for Distributed Deep Learning

This repository contains the implementation of different parallelism techniques for distributed training of GPT-2. It supports Data Parallel (DDP), Fully Sharded Data Parallel (FSDP), and Tensor Parallel (TP) training, as well as 2D parallelism combinations.

## Getting Started

### Hardware Requirements

* Multiple NVIDIA GPUs (minimum 2 for single parallelism, minimum 4 for 2D parallelism)

This code was developed and tested on:
- Development: A6000 48GB x 2 or A6000 48GB x 4 instances via [Prime Intellect](https://www.primeintellect.ai/) with PyTorch 2.5 CUDA 12.4 image (provided by Hyperstack cloud)
- Final evaluation: 8x A100 (40 GB SXM4) instance through Lambda Labs

### Installation

```bash
# Clone the repository
git clone https://github.com/rushilbhat/parallelism-experiments.git

# Navigate to the project directory
cd parallelism-experiments

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

Before training, you need to prepare the dataset:

```bash
# For Shakespeare dataset (small, prepares quickly)
python data/shakespeare/prepare.py

# For OpenWebText dataset (large, takes 20-30 minutes)
python data/openwebtext/prepare.py
```

The Shakespeare dataset is a toy dataset whose train and validation splits are generated instantly. OpenWebText is a heavy-duty dataset that takes approximately 20-30 minutes to create.

## Training

### Training Arguments

| Argument | Description |
|----------|-------------|
| `--tensor_parallel_size` | Degree of tensor parallelism (default: 1) |
| `--enable_loss_parallelism` | Enable loss parallelism with tensor parallel training |
| `--data_parallel_size` | Degree of data parallelism (default: 1) |
| `--data_parallel_type` | Choose data parallelisation strategy: ddp or fsdp |
| `--implementation` | Choose distributed implementation: custom or pytorch |
| `--deferred_init` | Delay materialisation of model parameters until sharding is applied |
| `--gradient_clipping` | Enable gradient clipping during training |
| `--eval_interval` | Interval between evaluations on validation set (default: 25) |
| `--dataset` | Choose dataset: shakespeare or openwebtext (default: shakespeare) |

### Launch Command

For distributed training, use `torchrun` to launch the training script across multiple GPUs:

```bash
torchrun --standalone --nproc_per_node=N train.py [arguments]
```
Example configurations:
| Configuration          | Arguments                                                                 |
|------------------------|---------------------------------------------------------------------------|
| DDP (2 GPUs)           | `--data_parallel_size=2 --data_parallel_type=ddp --implementation=custom` |
| FSDP (2 GPUs)          | `--data_parallel_size=2 --data_parallel_type=fsdp --implementation=custom --deferred_init` |
| TP (2 GPUs)            | `--tensor_parallel_size=2 --enable_loss_parallelism`                      |
| FSDP + TP (8 GPUs)     | `--tensor_parallel_size=2 --data_parallel_size=4 --data_parallel_type=fsdp --deferred_init --dataset=openwebtext` |


## Running Tests

Unit tests are included for each parallelism implementation and are designed to run on 2 GPUs. To run a specific test:

```bash
torchrun --standalone --nproc_per_node=2 -m unittest tests/[filename]
```

### Available Test Files

- `test_ddp.py` – Tests custom DDP vs PyTorch DDP
- `test_fsdp.py` – Tests custom FSDP vs PyTorch FSDP
- `test_tp.py` – Tests sharding consistency and numerical correctness of tensor parallelism 

## TensorBoard Logging

Training metrics are logged to TensorBoard. To view the logs when running on a remote server:

1. Start TensorBoard on the remote server:
```bash
tensorboard --logdir=runs --port=6006
```

2. Create an SSH tunnel from your local machine to the remote server:
```bash
ssh -L 6006:localhost:6006 user@remote-server
```

3. Open TensorBoard in your local browser:
```
http://localhost:6006
```

## Notes

- For optimal performance with FSDP, use the `--deferred_init` flag to delay parameter materialisation
- When using 2D parallelism (TP + DP), ensure that `tensor_parallel_size * data_parallel_size = num_gpus`
- Tensor parallelism with loss parallelism (`--enable_loss_parallelism`) typically provides better performance