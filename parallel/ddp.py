# ddp.py
import torch
import torch.nn as nn
import torch.distributed as dist

class Bucket:
    def __init__(self):
        self.parameters = {}
        self.size = 0 #in bytes
        self.pending_parameters = set()

    def add_param(self, named_param):
        name, param = named_param
        self.parameters[name] = param
        self.size += param.numel() * param.element_size()
        self.pending_parameters.add(param)

    def reset(self):
        self.pending_parameters = set(self.parameters.values())

class Reducer:
    def __init__(self, named_params, process_group, bucket_cap_mb):
        self.process_group = process_group
        self.bucket_cap_mb = bucket_cap_mb
        self.world_size = dist.get_world_size(self.process_group)
        self.buckets = []
        self.futures = []
        self._create_buckets(reversed(list(named_params)))
        self._register_hooks()
        self.require_backward_grad_sync = True

    def _create_buckets(self, named_params):
        current_bucket = Bucket()
        for name, param in named_params:
            if not param.requires_grad:
                continue
            current_bucket.add_param((name,param))
            if current_bucket.size > self.bucket_cap_mb * 1024 * 1024:
                self.buckets.append(current_bucket)
                current_bucket = Bucket()
        if current_bucket.size > 0:
            self.buckets.append(current_bucket)

    def _register_hooks(self):
        for bucket in self.buckets:
            for _, param in bucket.parameters.items():
                param.register_post_accumulate_grad_hook(lambda p, b=bucket: self._on_grad_ready(p, b))
    
    def _on_grad_ready(self, p, b):
        if not self.require_backward_grad_sync:
            return
        b.pending_parameters.remove(p)
        if not b.pending_parameters:
            self._reduce_bucket(b)

    def _reduce_bucket(self, bucket):
        flat_grads = torch.cat([param.grad.flatten() for _, param in bucket.parameters.items()])
        future = dist.all_reduce(flat_grads, group=self.process_group, op=dist.ReduceOp.AVG, async_op=True)
        self.futures.append((future, bucket))
        if len(self.futures) == len(self.buckets):
            self.finalize_backward()

    def finalize_backward(self):
        #Wait for all async reductions and update gradients
        for future, bucket in self.futures:
            future.wait()
            flat_grads = future.result()[0]
            self._unflatten_and_copy(flat_grads, bucket)

        self.futures.clear()
        for bucket in self.buckets:
            bucket.reset()

    def _unflatten_and_copy(self, flat_grads, bucket):
        offset = 0
        for _, param in bucket.parameters.items():
            numel = param.grad.numel()
            param.grad = flat_grads[offset:offset+numel].view_as(param)
            offset += numel

    
    def _measure_gpu_memory(self, stage):
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
        print(f"{stage}:")
        print(f"  Allocated: {memory_allocated:.2f} MB")
        print(f"  Reserved:  {memory_reserved:.2f} MB")


class CustomDDP(nn.Module):
    def __init__(self, module, process_group, bucket_cap_mb=25): 
        super().__init__()
        self.process_group = process_group
        self.world_size = dist.get_world_size(self.process_group)
        self.module = module
        self.reducer = Reducer(self.module.named_parameters(), process_group, bucket_cap_mb)

    def set_require_backward_grad_sync(self, require_sync):
        self.reducer.require_backward_grad_sync = require_sync

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    