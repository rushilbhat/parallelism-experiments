# train.py

# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
import os
import time
import math
import torch
import argparse
import functools
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from contextlib import nullcontext

from model import GPT, GPTConfig, Block
from data_loader import DataLoaderLite
from distributed import CustomDDP, CustomFSDP, clip_grad_norm
from tensor_parallel import apply_tensor_parallelism

def parse_args(is_distributed):
    parser = argparse.ArgumentParser(description='Training script with distributed options')
    if is_distributed:
        parser.add_argument('--tensor_parallel_size', type=int, default=1, 
                            help='Degree of tensor parallelism')
        parser.add_argument('--data_parallel_size', type=int, default=1, 
                            help='Degree of data parallelism')
        parser.add_argument('--data_parallel_type', type=str, choices=['ddp', 'fsdp'], 
                            help='Choose data parallelization strategy: ddp or fsdp')
        parser.add_argument('--implementation', type=str, choices=['pytorch', 'custom'], 
                            help='Choose distributed implementation: pytorch or custom')
        parser.add_argument('--deferred_init', action=argparse.BooleanOptionalAction, 
                            help="Delay materialisation of model parameters until sharding is applied")        
    parser.add_argument('--gradient_clipping', action=argparse.BooleanOptionalAction)
    return parser.parse_args()


# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
is_distributed = int(os.environ.get('RANK', -1)) != -1 # is this a distributed run?
if is_distributed:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    distributed_rank = int(os.environ['RANK'])
    distributed_local_rank = int(os.environ['LOCAL_RANK'])
    distributed_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{distributed_local_rank}'
    torch.cuda.set_device(device)
    master_process = distributed_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-distributed run
    distributed_rank = 0
    distributed_local_rank = 0
    distributed_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

args = parse_args(is_distributed)

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

HAS_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

# Set up DP/TP if distributed
if is_distributed:
    dp_size = args.data_parallel_size
    tp_size = args.tensor_parallel_size
    assert dp_size * tp_size == distributed_world_size, "dp_size * tp_size must equal distributed_world_size"

    tp_rank = distributed_rank % tp_size
    dp_rank = distributed_rank // tp_size

    tp_groups = [dist.new_group([tp + dp * tp_size for tp in range(tp_size)]) for dp in range(dp_size)]
    dp_groups = [dist.new_group([dp * tp_size + tp for dp in range(dp_size)]) for tp in range(tp_size)]
    tp_group = tp_groups[dp_rank]
    dp_group = dp_groups[tp_rank]

    if master_process:
        print(f"Running with DP size={dp_size} and TP size={tp_size}")
        print(f"This rank = {distributed_rank}, dp_rank={dp_rank}, tp_rank={tp_rank}")
else:
    dp_size = 1
    tp_size = 1
    dp_rank = 0
    tp_rank = 0
    tp_group = None
    dp_group = None

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 8 # micro batch size CHANGE
T = 1024 # sequence length
assert total_batch_size % (B * T * dp_size) == 0, "make sure total_batch_size is divisible by B * T * data parallel size"
grad_accum_steps = total_batch_size // (B * T * dp_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=dp_rank, num_processes=dp_size, master_process=master_process)

torch.set_float32_matmul_precision('high')

def get_param_dims(model):
    param_dims = {}
    for name, param in model.named_parameters():
        param_dims[name] = param.dim()
    return param_dims

device_context = torch.device('meta' if is_distributed and args.deferred_init else 'cpu')
with device_context:
    model = GPT(GPTConfig(vocab_size=50304))
param_dims = get_param_dims(model)

if is_distributed:
    if tp_size > 1:
        sharding_config = {
            'attn.c_attn1': 1, 'attn.c_attn2': 1, 'attn.c_attn3': 1,
            'attn.c_proj': 0,
            'mlp.c_fc': 1,
            'mlp.c_proj': 0,
        }
        apply_tensor_parallelism(model, 
                                 sharding_config, 
                                 tp_group, 
                                 reduce_row_output=True, 
                                 gather_col_output=False)
        
        if device_context.type == 'cpu':
            model.to(device)

    if dp_size > 1:
        if args.data_parallel_type == "fsdp":
            if args.implementation == "custom":
                model = CustomFSDP(model, process_group=dp_group, param_init_fn=model._init_weights)
            else: # pytorch
                def init_weights(module):
                    module.to_empty(device=torch.device(f'cuda:{distributed_local_rank}'), recurse=False)
                    model._init_weights(module)
                    
                    if module == model.lm_head and model.config.tie_word_embeddings:
                        model.transformer.wte.weight = module.weight

                gpt2_auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={
                        Block,
                    },
                )
                model = FSDP(model, process_group=dp_group, auto_wrap_policy=gpt2_auto_wrap_policy, use_orig_params=True, param_init_fn=init_weights)
        elif args.data_parallel_type == "ddp":
            if args.implementation == "custom":
                model = CustomDDP(model.to(device), dp_group)
            else: # pytorch
                model = DDP(model.to(device), device_ids=[dist.get_rank(dp_group)])

        if master_process: print(f"using {args.data_parallel_type} implementation: {args.implementation}")
else:
    model = model.to(device)

raw_model = model.module if dp_size > 1 else model # always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
scaler = torch.GradScaler(device_type, enabled=(not HAS_BF16))
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type, param_dims=param_dims, master_process=master_process)

# create the log directory we will write checkpoints to and log to
# log_dir = "log"
# os.makedirs(log_dir, exist_ok=True)
# log_file = os.path.join(log_dir, f"log.txt")
# with open(log_file, "w") as f: # open for writing to clear the file
#     pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    model.train()
    optimizer.zero_grad(set_to_none=not(dp_size > 1 and args.data_parallel_type == "fsdp" and args.implementation == "custom"))
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        grad_sync_context = nullcontext()
        if dp_size > 1:
            if args.data_parallel_type == 'ddp' and args.implementation == 'pytorch':
                grad_sync_context = model.no_sync() if micro_step < grad_accum_steps - 1 else grad_sync_context
            elif args.data_parallel_type == 'ddp' and args.implementation == 'custom':
                model.set_require_backward_grad_sync(micro_step == grad_accum_steps - 1)
        
        with grad_sync_context:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if HAS_BF16 else torch.float16):
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            scaler.scale(loss).backward()
    if dp_size > 1:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG, group=dp_group)

    if args.gradient_clipping:
        scaler.unscale_(optimizer)
        if is_distributed:
            norm = clip_grad_norm(model, max_norm=1.0, tp_group=tp_group, dp_group=dp_group, dp_type=args.data_parallel_type)
        # elif is_distributed and args.data_parallel_type == "fsdp":
        #     norm = model.clip_grad_norm_(max_norm=1.0)
        else:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    scaler.step(optimizer)
    scaler.update()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * distributed_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {f'{norm:.4f}' if args.gradient_clipping else None} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if is_distributed:
    destroy_process_group()