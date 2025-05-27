#fsdp.py
import torch
import torch.nn as nn
import torch.distributed as dist
from model import Block
from functools import reduce
from collections import deque
import math

class CustomFSDP(nn.Module):
    def __init__(self, module, process_group, param_init_fn):
        super().__init__()
        self.process_group = process_group
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)

        self._fsdp_wrapped_module = module

        self.param_numels = []
        self.param_shapes = []
        self.param_names = []

        self.flat_param = None
        self.local_shard = None

        self._wrap_blocks(self.module, param_init_fn)
        self._record_param_metadata()
        self._create_and_shard_flat_param(param_init_fn)

        self.register_full_backward_pre_hook(lambda m, go: self._pre_backward())
        self._register_post_backward_hooks()        
        
    def _wrap_blocks(self, module, param_init_fn):
        for name, child in module.named_children():
            if isinstance(child, Block):
                fsdp_unit = CustomFSDP(child, self.process_group, param_init_fn)
                setattr(module, name, fsdp_unit)
            else:
                self._wrap_blocks(child, param_init_fn)

    def _record_param_metadata(self):
        for n, p in self.module.named_parameters():
            if '_fsdp_wrapped_module' not in n:
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

        devices = {self.module.get_parameter(name).device for name in self.param_names}
        assert len(devices) == 1, "All parameters must be on the same device"
        is_materialised = (devices.pop() != torch.device('meta'))
        if is_materialised:
            offset = 0
            for name, numel in zip(self.param_names, self.param_numels):
                self.flat_param[offset:offset+numel] = self.module.get_parameter(name).data.view(-1)
                offset += numel
        else:
            self._materialise_params()
            self._update_module_params()

            def _apply_param_init_fn(root_module, param_init_fn):
                queue = deque([root_module])
                while queue:
                    module = queue.popleft()
                    if not isinstance(module, CustomFSDP):
                        param_init_fn(module)
                    for child in module.children():
                        if not isinstance(child, CustomFSDP):
                            queue.append(child)

    
            _apply_param_init_fn(self.module, param_init_fn)

        start_idx = self.rank * shard_size
        end_idx = start_idx + shard_size
        self.local_shard.data.add_(self.flat_param[start_idx: end_idx])
        self._shard()

    def _materialise_params(self):
        def _replace_param(param_path, new_param):
            *module_path, leaf = param_path.split('.')   
            submodule = reduce(getattr, module_path, self.module)
            setattr(submodule, leaf, new_param)

        for name in self.param_names:
            _replace_param(name, nn.Parameter(torch.empty(0, device='cuda')))
            
    def _update_module_params(self, include_grads=False):
        is_sharded = self.flat_param.data_ptr() == self.local_shard.data_ptr()
        local_shard_size = self.local_shard.numel()
        offset = 0 - local_shard_size * self.rank if is_sharded else 0

        for name, numel, shape in zip(self.param_names, self.param_numels, self.param_shapes):
            data_tensor, grad_tensor = self._retrieve_data_and_grad_tensors(offset, numel, shape, is_sharded, local_shard_size, include_grads)
            parameter = self.module.get_parameter(name)
            parameter.data = data_tensor
            if include_grads:
                parameter.grad = grad_tensor
            offset += numel

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
    
    def _gather(self, include_grads=False):
        full_tensor = torch.zeros(self.local_shard.numel() * self.world_size, device=self.local_shard.device)
        dist.all_gather_into_tensor(full_tensor, self.local_shard, group=self.process_group)
        self.flat_param.data = full_tensor

        if include_grads:
            full_grads_tensor = torch.zeros_like(self.flat_param)
            self.flat_param.grad = full_grads_tensor

        self._update_module_params(include_grads=include_grads)

    def _shard(self, include_grads=False):
        self.flat_param.data = self.local_shard.data
        if include_grads:
            self.flat_param.grad = self.local_shard.grad

        self._update_module_params(include_grads=include_grads)

    def _register_post_backward_hooks(self):
        for name in self.param_names:
            param = self._fsdp_wrapped_module.get_parameter(name)
            param.register_post_accumulate_grad_hook(lambda p: self._post_backward())

    def _pre_backward(self):
        if all(self._fsdp_wrapped_module.get_parameter(name).grad is None for name in self.param_names):
            self.local_shard.grad = torch.zeros_like(self.local_shard)
        self.grad_counter = 0
        self._gather(include_grads=True)
        
    def _post_backward(self):
        self.grad_counter += 1
        if self.grad_counter == len(list(self.param_names)):
            grad_shards = list(self.flat_param.grad.chunk(self.world_size))
            buffer = torch.empty(self.local_shard.shape, device='cuda')
            dist.reduce_scatter(buffer, grad_shards, op=dist.ReduceOp.AVG, group=self.process_group)
            self.local_shard.grad.add_(buffer)
            self._shard(include_grads=True)

    @property
    def module(self) -> nn.Module:
        return self._fsdp_wrapped_module

    def forward(self, *args, **kwargs):
        self._gather()
        output = self._fsdp_wrapped_module(*args, **kwargs)
        self._shard()
        return output
