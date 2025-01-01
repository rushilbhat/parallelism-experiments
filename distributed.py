# distributed.py
import torch
import torch.nn as nn
import torch.distributed as dist
import math
import gc
from model import Block
from functools import reduce

class Bucket:
    def __init__(self):
        self.parameters = {}
        self.size = 0 #in bytes
        self.grad_count = 0

    def add_param(self, named_param):
        name, param = named_param
        self.parameters[name] = param
        self.size += param.numel() * param.element_size()

class Reducer:
    def __init__(self, named_params, bucket_cap_mb, world_size):
        self.named_params = reversed(list(named_params))
        self.bucket_cap_mb = bucket_cap_mb
        self.world_size = world_size
        self.buckets = []
        self.futures = []
        self._create_buckets()
        self._register_hooks()
        self.require_backward_grad_sync = True
        self.reduced_buckets_count = 0

    def _create_buckets(self):
        current_bucket = Bucket()
        for name, param in self.named_params:
            if param.requires_grad:
                param_size = param.numel() * param.element_size() #using param_size as proxy for size of param.grad
                if current_bucket.size + param_size > self.bucket_cap_mb * 1024 * 1024:
                    self.buckets.append(current_bucket)
                    current_bucket = Bucket()
                current_bucket.add_param((name,param))
        self.buckets.append(current_bucket)

    def _register_hooks(self):
        # Add hooks to trigger gradient reduction when all grads in bucket are ready
        for bucket in self.buckets:
            for param in bucket.parameters.values():
                def hook(p, b=bucket):
                    if self.require_backward_grad_sync:
                        b.grad_count += 1
                        if b.grad_count == len(b.parameters):
                            self._reduce_bucket(b)
                            if self.reduced_buckets_count == len(self.buckets):
                                self.finalize_backward()

                param.register_post_accumulate_grad_hook(hook)

    def _reduce_bucket(self, bucket):
        flat_grads = torch.cat([param.grad.flatten() for param in bucket.parameters.values()])
        future = dist.all_reduce(flat_grads, async_op=True)
        self.futures.append((future, bucket))
        self.reduced_buckets_count += 1

    def finalize_backward(self):
        #Wait for all async reductions and update gradients
        for future, bucket in self.futures:
            future.wait()
            flat_grads = future.result()
            flat_grads[0].div_(self.world_size)
            self._unflatten_and_copy(flat_grads, bucket)

        self.futures.clear()
        self.reduced_buckets_count = 0

    def _unflatten_and_copy(self, flat_grads, bucket):
        offset = 0
        for name, param in bucket.parameters.items():
            numel = param.grad.numel()
            param.grad = flat_grads[0][offset:offset+numel].view_as(param)
            offset += numel
        bucket.grad_count = 0

    
    def _measure_gpu_memory(self, stage):
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
        print(f"{stage}:")
        print(f"  Allocated: {memory_allocated:.2f} MB")
        print(f"  Reserved:  {memory_reserved:.2f} MB")


class CustomDDP(nn.Module):
    def __init__(self, module, world_size, bucket_cap_mb=25): 
        super().__init__()
        self.module = module
        self.reducer = Reducer(self.module.named_parameters(), bucket_cap_mb, world_size)

    def set_require_backward_grad_sync(self, require_sync):
        self.reducer.require_backward_grad_sync = require_sync

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    

class CustomFSDP(torch.nn.Module):
    def __init__(self, module, param_init_fn, world_size, rank):
        super().__init__()
        self.world_size = world_size
        self.rank = rank

        self._fsdp_wrapped_module = module

        self.param_numels = []
        self.param_shapes = []
        self.param_names = []
        self.shared_params = {} # Maps shared parameter names to their original parameter names

        self.flat_param = None
        self.local_shard = None

        self._wrap_blocks(self.module, param_init_fn)
        self._record_param_metadata()

        self._create_and_shard_flat_param(param_init_fn)

        self.register_full_backward_pre_hook(lambda m, go: self._pre_backward())
        self._register_post_backward_hooks()        

    @property
    def module(self) -> nn.Module:
        return self._fsdp_wrapped_module
        
    def _wrap_blocks(self, module, param_init_fn):
        for name, child in module.named_children():
            if isinstance(child, Block):
                fsdp_unit = CustomFSDP(child, param_init_fn, self.world_size, self.rank)
                setattr(module, name, fsdp_unit)
            else:
                self._wrap_blocks(child, param_init_fn)

    def _record_param_metadata(self):
        all_params = {n: p for n, p in self.module.named_parameters(remove_duplicate=False) if '_fsdp_wrapped_module' not in n}
        unique_params = {n: p for n, p in self.module.named_parameters() if '_fsdp_wrapped_module' not in n}

        # Record shared parameters
        if len(all_params) > len(unique_params):
            param_id_to_name = {id(p): n for n, p in unique_params.items()}
            self.shared_params.update({n: param_id_to_name[id(p)] for n, p in all_params.items() 
                                     if param_id_to_name.get(id(p)) and param_id_to_name[id(p)] != n})

        # Record parameter metadata
        for n, p in unique_params.items():
            self.param_numels.append(p.numel())
            self.param_shapes.append(p.shape) 
            self.param_names.append(n)

    def _create_and_shard_flat_param(self, param_init_fn):
        total_numel = sum(self.param_numels)
        padded_size = math.ceil(total_numel / self.world_size) * self.world_size
        shard_size = padded_size // self.world_size

        self.flat_param = torch.zeros(padded_size, device='cuda')
        self.local_shard = torch.zeros(shard_size, device='cuda')
        self.local_shard.grad = torch.zeros_like(self.local_shard)
        self._update_module_params(include_grads=False)

        def apply_param_init_fn(module, param_init_fn):
            if not isinstance(module, CustomFSDP):
                param_init_fn(module)
            
            for child in module.children():
                if not isinstance(child, CustomFSDP):
                    apply_param_init_fn(child, param_init_fn)
 
        apply_param_init_fn(self.module, param_init_fn)

        start_idx = self.rank * shard_size
        end_idx = start_idx + shard_size
        self.local_shard.data.add_(self.flat_param[start_idx: end_idx])
        self._shard()


    def _retrieve_data_and_grad_tensors(self, offset, numel, shape, is_sharded, local_shard_size, include_grads):
        if is_sharded:
            # Handle cases where parameter lies outside this shard
            if offset + numel < 0 or offset >= local_shard_size:
                return torch.empty(0, device='cuda'), None
            
            # Get slice of parameter from local shard
            start = max(offset, 0)
            end = min(offset + numel, local_shard_size)
            data_tensor = self.local_shard[start:end]
            grad_tensor = self.local_shard.grad[start:end] if include_grads else None
        else:
            # Get slice from full flattened parameter
            data_tensor = self.flat_param[offset:offset+numel].view(shape)
            grad_tensor = self.flat_param.grad[offset:offset+numel].view(shape) if include_grads else None
        
        return data_tensor, grad_tensor

    def _assign_sliced_tensors_to_param(self, param_name, data_tensor, grad_tensor, include_grads):
        parameter = self.module.get_parameter(param_name)

        if parameter.device.type == 'meta':
            self.module.to_empty(device="cuda")
            parameter = self.module.get_parameter(param_name)

        parameter.data = data_tensor
        if include_grads:
            parameter.grad = grad_tensor

    def _handle_shared_params(self):
        if len(self.shared_params) > 0:
            unique_params = dict(self.module.named_parameters())
            for duplicate, original in self.shared_params.items():
                *module_path, leaf = duplicate.split('.')
                submodule = reduce(getattr, module_path, self.module)
                setattr(submodule, leaf, unique_params[original])
    
    def _update_module_params(self, include_grads, flag=False):
        is_sharded = self.flat_param.data_ptr() == self.local_shard.data_ptr()
        local_shard_size = self.local_shard.numel()
        offset = 0 - local_shard_size * self.rank if is_sharded else 0

        # Update all parameters
        for name, numel, shape in zip(self.param_names, self.param_numels, self.param_shapes):
            data_tensor, grad_tensor = self._retrieve_data_and_grad_tensors(offset, numel, shape, is_sharded, local_shard_size, include_grads)
            self._assign_sliced_tensors_to_param(name, data_tensor, grad_tensor, include_grads)
            offset += numel

        self._handle_shared_params()

    def _gather(self, include_grads=False, flag=True):
        full_tensor = torch.zeros(self.local_shard.numel() * self.world_size, device=self.local_shard.device)
        dist.all_gather_into_tensor(full_tensor, self.local_shard)
        self.flat_param.data = full_tensor

        if include_grads:
            full_grads_tensor = torch.zeros(self.local_shard.grad.numel() * self.world_size, device=self.local_shard.grad.device)
            self.flat_param.grad = full_grads_tensor

        self._update_module_params(include_grads=include_grads)


    def _shard(self, include_grads=False, flag=False):
        self.flat_param.data = self.local_shard.data
        if include_grads:
            self.flat_param.grad = self.local_shard.grad

        self._update_module_params(include_grads=include_grads)

    def _pre_backward(self):
        self.grad_counter = 0
        self._gather(include_grads=True)
        
    def _post_backward(self):
        self.grad_counter += 1
        if self.grad_counter == len(list(self.param_names)):
            grad_shards = list(self.flat_param.grad.chunk(self.world_size))
            buffer = torch.empty(self.local_shard.shape, device='cuda')
            dist.reduce_scatter(buffer, grad_shards, op=dist.ReduceOp.AVG)
            self.local_shard.grad.add_(buffer)
            self._shard(include_grads=True)

    def _register_post_backward_hooks(self):
        for name in self.param_names:
            param = dict(self._fsdp_wrapped_module.named_parameters())[name]
            param.register_post_accumulate_grad_hook(lambda p: self._post_backward())



    @torch.no_grad()
    def clip_grad_norm_(self, max_norm):
        local_grad_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(p.grad, 2.0, dtype=torch.float32) 
                    for p in self._fsdp_wrapped_module.parameters()
                    if p.grad is not None
                ]
            ),
            2.0,
            dtype=torch.float32
        )
        sqaured_norm = local_grad_norm ** 2.0
        dist.all_reduce(sqaured_norm) 
        global_grad_norm = sqaured_norm ** (1.0 / 2.0)
        clip_coef = max_norm / (global_grad_norm + 1e-6)
        if clip_coef < 1.0:
            for p in self.module.parameters():
                if p.grad is not None:
                    p.grad.mul_(clip_coef)
        return global_grad_norm

            
    def forward(self, *args, **kwargs):
        self._gather()
        output = self._fsdp_wrapped_module(*args, **kwargs)
        self._shard()
        return output