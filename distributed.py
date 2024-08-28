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


class FSDPUnit:
    def __init__(self, module_list, param_init_fn, world_size, rank, unit_name):
        self.module_list = module_list
        self.world_size = world_size
        self.rank = rank
        self.is_master = (rank == 0)
        self.shard = None
        self.flat_param = None
        self.param_numels = []
        self.param_shapes = []
        self.param_names = []
        self.shared_params = {}
        self.unit_name = unit_name
        self.before_buffer = 0
        self.after_buffer = 0

        self._record_param_metadata()
        self._create_and_shard_flat_param(param_init_fn)

        torch.cuda.empty_cache()
        gc.collect()
        if self.is_master: self._measure_gpu_memory("After sharding flat_param")
        
    def _measure_gpu_memory(self, stage, before_flag=False, after_flag=False):
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
        print(f"[Rank {self.rank}] {stage}:")
        print(f"  Allocated: {memory_allocated:.2f} MB")
        print(f"  Reserved:  {memory_reserved:.2f} MB")
        if before_flag: self.before_buffer = memory_allocated
        if after_flag: self.after_buffer = memory_allocated

    def _record_param_metadata(self):
        all_params = dict(self.module_list.named_parameters(remove_duplicate=False))
        unique_params = dict(self.module_list.named_parameters())
        param_id_to_name = {id(param): name for name, param in unique_params.items()}
        
        if len(all_params) > len(unique_params):
            for name, param in all_params.items():
                unique_name = param_id_to_name.get(id(param))
                if unique_name is not None and unique_name != name:
                    self.shared_params[name] = unique_name

        for n,p in unique_params.items():
            self.param_numels.append(p.numel())
            self.param_shapes.append(p.shape)
            self.param_names.append(n)


    def _create_and_shard_flat_param(self, param_init_fn):
        total_numel = sum(self.param_numels)
        padded_size = math.ceil(total_numel / self.world_size) * self.world_size
        shard_size = padded_size // self.world_size
        if self.is_master: self._measure_gpu_memory("Before creating shard")
        self.shard = nn.Parameter(torch.empty(shard_size, device='cuda'))
        self.shard.grad = torch.zeros_like(self.shard)
        if self.is_master: self._measure_gpu_memory("After creating shard")
        
        with torch.no_grad():
            if self.is_master:
                self.flat_param = nn.Parameter(torch.zeros(padded_size, device='cuda'))
                self.flat_param.grad = torch.zeros_like(self.flat_param)  
                if self.is_master: self._measure_gpu_memory("After creating flat_param")
                self._assign_params_and_grads()
                for m in self.module_list.modules():
                    if len(list(m.children())) == 0 and hasattr(m, 'reset_parameters'): 
                        m.reset_parameters() #doesn't break weight sharing scheme since it's an in place operation
                self.module_list.apply(param_init_fn)
                flat_param_shards = list(self.flat_param.chunk(self.world_size))
                flat_grad_shards = list(self.flat_param.grad.chunk(self.world_size))

            dist.scatter(self.shard.data, flat_param_shards if self.is_master else None, src=0)
            dist.scatter(self.shard.grad, flat_grad_shards if self.is_master else None, src=0)
        
        self.flat_param = nn.Parameter(self.shard)
        self.flat_param.grad = self.shard.grad
        self._assign_params_and_grads()

    def _assign_params_and_grads(self):
        is_sharded = self.flat_param.data_ptr() == self.shard.data_ptr()
        offset = 0
        for name, numel, shape in zip(self.param_names, self.param_numels, self.param_shapes):
            if is_sharded:
                param_tensor = torch.empty(0, device='cuda')
                grad_tensor = torch.empty(0, device='cuda')
            else:
                param_tensor = self.flat_param[offset:offset+numel].view(shape)
                grad_tensor = self.flat_param.grad[offset:offset+numel].view(shape)
            name_parts = name.split('.')
            module = self.module_list #clean up naming
            for part in name_parts[:-1]:
                module = getattr(module, part)
            if getattr(module, name_parts[-1]).device.type == 'meta':
                setattr(module, name_parts[-1], nn.Parameter(param_tensor))
            else:
                getattr(module, name_parts[-1]).data = param_tensor
            getattr(module, name_parts[-1]).grad = grad_tensor
            offset += numel

        unique_params = dict(self.module_list.named_parameters())
        if len(self.shared_params) > 0:
            for duplicate, original in self.shared_params.items():
                name_parts = duplicate.split('.')
                module = self.module_list
                for part in name_parts[:-1]:
                    module = getattr(module, part)
                setattr(module, name_parts[-1], unique_params[original])

    def gather_params(self):
        if self.flat_param.data_ptr() == self.shard.data_ptr(): #check if all shards have been gathered
            if self.is_master: self._measure_gpu_memory(f"Before gathering unit {self.unit_name}", before_flag=True)
            full_tensor = torch.empty(self.shard.numel() * self.world_size, device=self.shard.device)
            full_grads_tensor = torch.empty(self.shard.grad.numel() * self.world_size, device=self.shard.grad.device)
            if self.is_master: self._measure_gpu_memory(f"After creating full tensor for {self.unit_name}")
            dist.all_gather_into_tensor(full_tensor, self.shard)
            dist.all_gather_into_tensor(full_grads_tensor, self.shard.grad)
            self.flat_param.data = full_tensor
            self.flat_param.grad = full_grads_tensor

            self._assign_params_and_grads()
            if self.is_master: 
                self._measure_gpu_memory(f"After gathering unit {self.unit_name}", after_flag=True)
                print(f"Increase: {self.after_buffer - self.before_buffer}")


            
    def shard_params(self):
        if self.flat_param.data_ptr() != self.shard.data_ptr(): #check if flat_params is not sharded
            if self.is_master: self._measure_gpu_memory(f"Before sharding unit {self.unit_name}", before_flag=True)
            self.flat_param.data = self.shard.data
            self.flat_param.grad = self.shard.grad
            self._assign_params_and_grads()

            torch.cuda.empty_cache()
            gc.collect()
            if self.is_master: 
                self._measure_gpu_memory(f"After sharding unit {self.unit_name}", after_flag=True)
                print(f"Decrease: {self.before_buffer - self.after_buffer}")


class CustomFSDP(nn.Module):
    def __init__(self, module, param_init_fn, world_size, rank):
        super().__init__()
        self.module = module
        self.world_size = world_size
        self.rank = rank
        self.fsdp_units = self._create_fsdp_units_for_gpt(self.module, param_init_fn)
        self._register_hooks()

        
    def _create_fsdp_units_for_gpt(self, gpt_model, param_init_fn):
        fsdp_units = []
        for block in gpt_model.transformer.h:
            fsdp_units.append(FSDPUnit(nn.ModuleList([block]), param_init_fn, self.world_size, self.rank, "block"))
        remaining_params = nn.ModuleList([
            gpt_model.transformer.wpe,
            gpt_model.transformer.wte,
            gpt_model.transformer.ln_f, 
            gpt_model.lm_head
        ])
        fsdp_units.append(FSDPUnit(remaining_params, param_init_fn, self.world_size, self.rank, "remaining"))
        return fsdp_units
    
    def _register_hooks(self):
        for i, unit in enumerate(self.fsdp_units):
            for j, module in enumerate(unit.module_list):
                if j == 0:
                    module.register_forward_pre_hook(lambda m, i, u=unit: u.gather_params())
                # Only add post-forward hook to the last module in each unit except for the last fsdp unit 
                # (wrt declared order, not logical order) 
                # CHANGE
                if j == len(unit.module_list)-1:
                    module.register_forward_hook(lambda m, i, o, u=unit: u.shard_params())
                
    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        return output