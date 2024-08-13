# distributed.py
import torch
import torch.nn as nn
import torch.distributed as dist
import math
import gc

class Bucket:
    def __init__(self):
        self.parameters = {}
        self.gradients = {}
        self.size = 0 #in bytes
        self.grad_count = 0

    def add_param(self, named_param):
        name, param = named_param
        self.parameters[name] = param
        self.size += param.numel() * param.element_size()
    
    def add_grad(self, named_grad):
        name, grad = named_grad
        self.gradients[name] = grad

    def reset(self):
        self.gradients.clear()


class CustomDDP(nn.Module):
    def __init__(self, module, process_group, bucket_cap_mb=25): 
        super().__init__()
        self.module = module
        self.process_group = process_group
        self.bucket_cap_mb = bucket_cap_mb
        self.buckets = []
        self.futures = []
        self.require_backward_grad_sync = True
        self._create_buckets()
        self._register_hooks()

    def _create_buckets(self):
        named_params = reversed(list(self.module.named_parameters()))
                
        current_bucket = Bucket()

        for name, param in named_params:
            if param.requires_grad:
                param_size = param.numel() * param.element_size() #using param_size as proxy for size of param.grad
                if current_bucket.size + param_size > self.bucket_cap_mb * 1024 * 1024:
                    self.buckets.append(current_bucket)
                    current_bucket = Bucket()
                current_bucket.add_param((name,param))
        self.buckets.append(current_bucket)
        
    def _create_hook(self, bucket, name, param):
        def hook(grad):
            if self.require_backward_grad_sync:
                accumulated_grad = param.grad + grad
                bucket.add_grad((name, accumulated_grad))
                if len(bucket.gradients) == len(bucket.parameters):
                    self._reduce_bucket(bucket)
        return hook

    def _register_hooks(self):
        for bucket in self.buckets:
            for name, param in bucket.parameters.items():
                hook = self._create_hook(bucket, name, param)
                param.register_hook(hook)

    def _reduce_bucket(self, bucket):
        flat_grads = torch.cat([grad.flatten() for grad in bucket.gradients.values()])
        future = dist.all_reduce(flat_grads, group=self.process_group, async_op=True)
        self.futures.append((future, bucket))

    def set_require_backward_grad_sync(self, require_sync):
        self.require_backward_grad_sync = require_sync

    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finalize_backward(self):
        world_size = dist.get_world_size(self.process_group)
        for future, bucket in self.futures:
            future.wait()
            flat_grads = future.result()
            flat_grads[0].div_(world_size)
            self._unflatten_and_copy(flat_grads, bucket)
        self.futures.clear()

    def _unflatten_and_copy(self, flat_grads, bucket):
        offset = 0
        for name, grad in bucket.gradients.items():
            numel = grad.numel()
            if name in bucket.parameters:
                param = bucket.parameters[name]
                param.grad = flat_grads[0][offset:offset+numel].view_as(grad)
                offset += numel
        bucket.reset()

class CustomFSDP(nn.Module):
    def __init__(self, module, param_init_fn, world_size, rank):
        super().__init__()
        self.module = module
        self.param_init_fn = param_init_fn  
        self.world_size = world_size
        self.rank = rank
        self.is_master = (rank == 0)
        self.fsdp_units = self._create_fsdp_units_for_gpt(self.module)
        self.shards = []
        self.unsharded_fsdp_units = {}

        for fsdp_unit in self.fsdp_units:
            shard = self._create_and_shard_flat_param(fsdp_unit)
            self.shards.append(shard)
        
    def _create_fsdp_units_for_gpt(self, gpt_model):
        fsdp_units = []
        for block in gpt_model.transformer.h:
            fsdp_units.append(nn.ModuleList([block]))
        remaining_params = nn.ModuleList([
            gpt_model.transformer.wte, 
            gpt_model.transformer.wpe, 
            gpt_model.transformer.ln_f, 
            gpt_model.lm_head
        ])
        fsdp_units.append(remaining_params)

        return fsdp_units
    
    def _create_and_shard_flat_param(self, fsdp_unit):
        total_numel = sum(p.numel() for p in fsdp_unit.parameters())
        padded_size = math.ceil(total_numel / self.world_size) * self.world_size
        shard_size = padded_size // self.world_size
        
        if self.is_master:
            with torch.no_grad():
                flat_param = nn.Parameter(torch.zeros(padded_size, device='cuda'))
                self._assign_params(fsdp_unit, flat_param)            
                fsdp_unit.apply(self.param_init_fn)
                flat_param_shards = list(flat_param.chunk(self.world_size))   

        shard = torch.empty(shard_size, device='cuda')
        dist.scatter(shard, flat_param_shards if self.is_master else None, src=0)

        if self.is_master:
            flat_param_meta = nn.Parameter(torch.empty(padded_size, device='meta'))
            self._assign_params(fsdp_unit, flat_param_meta)

            del flat_param
            del flat_param_shards
            torch.cuda.empty_cache()
            gc.collect()

        return shard
    
    def _assign_params(self, fsdp_unit, flat_param):
        all_params = dict(fsdp_unit.named_parameters(remove_duplicate=False))
        unique_params = dict(fsdp_unit.named_parameters())
        param_id_to_name = {id(param): name for name, param in unique_params.items()}
        
        shared_params = {}
        if len(all_params) > len(unique_params):
            for name, param in all_params.items():
                unique_name = param_id_to_name.get(id(param))
                if unique_name is not None and unique_name != name:
                    shared_params[name] = unique_name

        offset = 0
        for name, param in unique_params.items():
            param_shape = param.shape
            param_numel = param.numel()
            param_view = flat_param.data[offset:offset+param.numel()].view(param_shape)
            name_parts = name.split('.')
            module = fsdp_unit
            for part in name_parts[:-1]:
                module = getattr(module, part)
            setattr(module, name_parts[-1], nn.Parameter(param_view))
            offset += param_numel
        unique_params = dict(fsdp_unit.named_parameters())

        for shared_name, unique_name in shared_params.items():
            name_parts = shared_name.split('.')
            module = fsdp_unit
            for part in name_parts[:-1]:
                module = getattr(module, part)
            setattr(module, name_parts[-1], unique_params[unique_name])

