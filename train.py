# train.py
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import time
import math
import torch
import torch.nn.functional as F
import argparse
import functools
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from contextlib import nullcontext

from model import GPT, GPTConfig, Block
from parallel.ddp import CustomDDP
from parallel.fsdp import CustomFSDP
from parallel.utils import clip_grad_norm
from parallel.tensor_parallel import (
    apply_tensor_parallelism, 
    vocab_parallel_cross_entropy_loss
)
from torch.utils.tensorboard import SummaryWriter

torch.use_deterministic_algorithms(True)

def parse_args(is_distributed):
    parser = argparse.ArgumentParser(description='Training script with distributed options')
    if is_distributed:
        parser.add_argument('--tensor_parallel_size', type=int, default=1, 
                            help='Degree of tensor parallelism')
        parser.add_argument('--enable_loss_parallelism', action=argparse.BooleanOptionalAction, 
                            help='Enable loss parallelism')
        parser.add_argument('--data_parallel_size', type=int, default=1, 
                            help='Degree of data parallelism')
        parser.add_argument('--data_parallel_type', type=str, choices=['ddp', 'fsdp'], 
                            help='Choose data parallelization strategy: ddp or fsdp')
        parser.add_argument('--implementation', type=str, choices=['pytorch', 'custom'], 
                            help='Choose distributed implementation: pytorch or custom')
        parser.add_argument('--deferred_init', action=argparse.BooleanOptionalAction, 
                            help="Delay materialisation of model parameters until sharding is applied")        
    parser.add_argument('--gradient_clipping', action=argparse.BooleanOptionalAction)
    parser.add_argument('--eval_interval', type=int, default=25, 
                        help='Interval between evaluations on validation set')
    parser.add_argument('--dataset', type=str, default='shakespeare',
                        help='Choose dataset: shakespeare or openwebtext')
    return parser.parse_args()


class Trainer:
    def __init__(self):
        self.is_distributed = int(os.environ.get('RANK', -1)) != -1
        self.args = parse_args(self.is_distributed)

        self._init_distributed()
        
        self.device = self._select_device()
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        torch.manual_seed(1337)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1337)

        self.total_batch_size = 524288  # in tokens
        self.B = 8      # micro batch size
        self.T = 1024   # sequence length
        self._compute_grad_accum_steps()

        self.data_dir = f'data/{self.args.dataset}'   
        # Check if train and val bin files exist
        train_path = os.path.join(self.data_dir, 'train.bin')
        val_path = os.path.join(self.data_dir, 'val.bin')
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at {train_path}. Run prepare.py in {self.data_dir} directory first.")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found at {val_path}. Run prepare.py in {self.data_dir} directory first.")

        self.eval_interval = self.args.eval_interval
        if self.master_process:
            log_dir = 'runs/'
            for arg_name, arg_value in vars(self.args).items():
                log_dir += f'{arg_name}={arg_value}_'
            self.writer = SummaryWriter(log_dir=log_dir)

        self.model, self.param_dims = self._build_model()

        # Mixed-precision / BF16
        self.HAS_BF16 = torch.cuda.is_available() and (torch.cuda.get_device_capability()[0] >= 8)
        self.scaler = torch.GradScaler(self.device_type, enabled=(not self.HAS_BF16))

        # Create optimizer
        self.optimizer = self.model.module.configure_optimizers(
            weight_decay=0.1,
            learning_rate=6e-4,
            device_type=self.device_type,
            param_dims=self.param_dims,
            master_process=self.master_process,
        ) if self.dp_size > 1 else self.model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=6e-4,
            device_type=self.device_type,
            param_dims=self.param_dims,
            master_process=self.master_process,
        )

        # Learning rate schedule
        self.max_lr = 6e-4
        self.min_lr = self.max_lr * 0.1
        self.warmup_steps = 10
        self.max_steps = 250

    def _init_distributed(self):
        if self.is_distributed:
            init_process_group(backend='nccl')
            self.distributed_rank = int(os.environ['RANK'])
            self.distributed_local_rank = int(os.environ['LOCAL_RANK'])
            self.distributed_world_size = int(os.environ['WORLD_SIZE'])
            self.master_process = (self.distributed_rank == 0)
        else:
            self.distributed_rank = 0
            self.distributed_local_rank = 0
            self.distributed_world_size = 1
            self.master_process = True

        if self.is_distributed:
            assert torch.cuda.is_available()
            torch.cuda.set_device(f'cuda:{self.distributed_local_rank}')
        
        if self.is_distributed:
            self.dp_size = self.args.data_parallel_size
            self.tp_size = self.args.tensor_parallel_size
            assert self.dp_size * self.tp_size == self.distributed_world_size, \
                "dp_size * tp_size must match world_size"

            self.tp_rank = self.distributed_rank % self.tp_size
            self.dp_rank = self.distributed_rank // self.tp_size

            self.tp_groups = [
                dist.new_group([tp + dp * self.tp_size for tp in range(self.tp_size)]) 
                for dp in range(self.dp_size)
            ]
            self.dp_groups = [
                dist.new_group([dp * self.tp_size + tp for dp in range(self.dp_size)]) 
                for tp in range(self.tp_size)
            ]
            self.tp_group = self.tp_groups[self.dp_rank]
            self.dp_group = self.dp_groups[self.tp_rank]

            if self.master_process:
                print(f"Running with DP size={self.dp_size}, TP size={self.tp_size}")
                print(f"This rank={self.distributed_rank}, dp_rank={self.dp_rank}, tp_rank={self.tp_rank}")
        else:
            self.dp_size = 1
            self.tp_size = 1
            self.dp_rank = 0
            self.tp_rank = 0
            self.tp_group = None
            self.dp_group = None

    def _select_device(self):
        if self.is_distributed:
            return f"cuda:{self.distributed_local_rank}"
        else:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            if self.master_process:
                print(f"Using device: {device}")
            return device
    
    def _compute_grad_accum_steps(self):
        self.grad_accum_steps = self.total_batch_size // (self.B * self.T * self.dp_size)
        if self.master_process:
            print(f"Total desired batch size: {self.total_batch_size}")
            print(f"=> calculated gradient accumulation steps: {self.grad_accum_steps}")
    
    def _build_model(self):
        """Build and wrap the GPT model with parallel strategies if needed."""
        device_context = torch.device('meta') if self.is_distributed and self.args.deferred_init else torch.device(self.device)
        with device_context:
            model = GPT(GPTConfig(vocab_size=50304))

        param_dims = self._get_param_dims(model)

        # Tensor Parallel
        if self.tp_size > 1:
            apply_tensor_parallelism(
                model,
                self.tp_group,
                enable_loss_parallelism=self.args.enable_loss_parallelism
            )

            # Need to initialise params if on a meta device and will not be wrapped in FSDP next.
            if (device_context.type == 'meta' 
                and not (self.dp_size > 1 and self.args.data_parallel_type == 'fsdp')):
                def init_weights(module):
                    module.to_empty(device=torch.device(f'cuda:{self.distributed_local_rank}'), recurse=False)
                    model._init_weights(module)
                model.apply(init_weights)

        # Data Parallel
        if self.dp_size > 1:
            if self.args.data_parallel_type == "fsdp":
                model = self.wrap_fsdp(model)
            elif self.args.data_parallel_type == "ddp":
                model = self._wrap_ddp(model)
            
            if self.master_process:
                print(f"Using {self.args.data_parallel_type} implementation: {self.args.implementation}")
        else:
            model = model.to(self.device)

        return model, param_dims

    def wrap_fsdp(self, model):
        """FSDP wrapping, either custom or PyTorch."""
        if self.args.implementation == "custom":
            return CustomFSDP(
                model, 
                process_group=self.dp_group, 
                param_init_fn=model._init_weights
            )
        else:
            gpt2_auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Block},
            )
            def init_weights(module):
                module.to_empty(device=torch.device(f'cuda:{self.distributed_local_rank}'), recurse=False)
                model._init_weights(module)
            return FSDP(
                model, 
                process_group=self.dp_group, 
                auto_wrap_policy=gpt2_auto_wrap_policy, 
                use_orig_params=True, 
                param_init_fn=init_weights
            )

    def _wrap_ddp(self, model):
        """DDP wrapping, either custom or PyTorch."""
        if self.args.implementation == "custom":
            return CustomDDP(model.to(self.device), self.dp_group)
        else:
            return DDP(model.to(self.device), device_ids=[dist.get_rank(self.dp_group)])
    
    @staticmethod
    def _get_param_dims(model):
        param_dims = {}
        for name, param in model.named_parameters():
            param_dims[name] = param.dim()
        return param_dims

    def _get_lr(self, step):
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        if step > self.max_steps:
            return self.min_lr
        # Cosine decay
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def get_batch(self, split):
        if split == 'train':
            data = np.memmap(os.path.join(self.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(self.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - self.T, (self.B,))
        x = torch.stack([torch.from_numpy((data[i:i+self.T]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.T]).astype(np.int64)) for i in ix])
        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def evaluate(self):
        self.model.eval()
        eval_iters = 10 
        losses = torch.zeros(eval_iters, device=self.device)
        
        with torch.no_grad():
            for k in range(eval_iters):
                x, y = self.get_batch('val')
                with torch.autocast(device_type=self.device_type, 
                                    dtype=torch.bfloat16 if self.HAS_BF16 else torch.float16):
                    logits = self.model(x)
                    if self.tp_size > 1 and self.args.enable_loss_parallelism:
                        loss = vocab_parallel_cross_entropy_loss(logits, y, self.tp_group)
                    else:
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    losses[k] = loss.item()
        
        val_loss = losses.mean()
        
        # Average across distributed processes if using DP
        if self.dp_size > 1:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG, group=self.dp_group)
            
        self.model.train()
        return val_loss

    def train(self):
        """Main training loop."""
        torch.set_float32_matmul_precision('high')
        for step in range(self.max_steps):
            t0 = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            loss_accum = 0.0

            if step % self.eval_interval == 0:
                val_loss = self.evaluate()        
                if self.master_process:
                    self.writer.add_scalar('Loss/val', val_loss.item(), step)
                    print(f"EVAL {step:5d} | val loss: {val_loss:.6f}")

            for micro_step in range(self.grad_accum_steps):
                x, y = self.get_batch('train')
                grad_sync_context = self._no_sync_context(micro_step)

                with grad_sync_context:
                    with torch.autocast(device_type=self.device_type, 
                                        dtype=torch.bfloat16 if self.HAS_BF16 else torch.float16):
                        logits = self.model(x)
                        if self.tp_size > 1 and self.args.enable_loss_parallelism:
                            loss = vocab_parallel_cross_entropy_loss(logits, y, self.tp_group)
                        else:
                            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

                    # Average loss across micro-steps
                    loss = loss / self.grad_accum_steps
                    loss_accum += loss.detach()
                    self.scaler.scale(loss).backward()

            # Average the loss across DP ranks
            if self.dp_size > 1:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG, group=self.dp_group)

            # Optional gradient clipping
            norm = None
            if self.args.gradient_clipping:
                self.scaler.unscale_(self.optimizer)
                norm = self._clip_grad()

            # Set LR and step the optimizer
            lr = self._get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Timing / stats
            if self.device_type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            tokens_processed = self.B * self.T * self.grad_accum_steps * self.distributed_world_size
            tokens_per_sec = tokens_processed / dt

            if self.master_process:
                self.writer.add_scalar('Loss/train', loss_accum.item(), step)
                if norm is not None: 
                    self.writer.add_scalar('GradNorm/train', norm.item(), step)
                print(
                    f"step {step:5d} | loss: {loss_accum.item():.6f} | "
                    f"lr {lr:.4e} | "
                    f"norm: {f'{norm:.4f}' if self.args.gradient_clipping else None} | "
                    f"dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
                )

        # Cleanup
        if self.master_process:
            self.writer.close()
        if self.is_distributed:
            destroy_process_group()

    def _no_sync_context(self, micro_step):
        """Return the right no_sync context for the current DP approach."""
        if self.dp_size > 1 and self.args.data_parallel_type == 'ddp':
            if self.args.implementation == 'pytorch':
                if micro_step < self.grad_accum_steps - 1:
                    return self.model.no_sync()
            elif self.args.implementation == 'custom':
                self.model.set_require_backward_grad_sync(micro_step == self.grad_accum_steps - 1)
        return nullcontext()

    def _clip_grad(self):
        if self.is_distributed:
            return clip_grad_norm(
                self.model,
                max_norm=1.0,
                tp_group=self.tp_group,
                dp_group=self.dp_group,
                dp_type=self.args.data_parallel_type,
            )
        else:
            return torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()