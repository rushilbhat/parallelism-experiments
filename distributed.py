# distributed.py
import torch
import torch.nn as nn
import torch.distributed as dist
import math
import gc

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
    def __init__(self, module, process_group, bucket_cap_mb=25): 
        super().__init__()
        self.module = module
        self.process_group = process_group
        world_size = dist.get_world_size(self.process_group)
        self.reducer = Reducer(self.module.named_parameters(), bucket_cap_mb, world_size)

    def set_require_backward_grad_sync(self, require_sync):
        self.reducer.require_backward_grad_sync = require_sync

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    

class FSDPUnit:
    def __init__(self, module_list, param_init_fn, world_size, rank, unit_name):
        self.module_list = module_list
        self.world_size = world_size
        self.rank = rank
        self.is_master = (rank == 0)
        self.local_shard = None
        self.flat_param = None
        self.param_numels = []
        self.param_shapes = []
        self.param_names = []
        self.shared_params = {}
        self.unit_name = unit_name
        self.before_buffer = 0
        self.after_buffer = 0
        self.grad_counter = 0

        self._record_param_metadata()
        self._create_and_shard_flat_param(param_init_fn)

        torch.cuda.empty_cache()
        gc.collect()

        # if self.is_master: self._measure_gpu_memory(f"After sharding flat_param and grads")

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
        # if self.is_master: self._measure_gpu_memory(f"Before creating shard ")
        self.local_shard = nn.Parameter(torch.empty(shard_size, device='cuda'))
        self.local_shard.grad = torch.zeros_like(self.local_shard)
        # if self.is_master: self._measure_gpu_memory(f"After creating shard and grad ")
        
        with torch.no_grad():
            if self.is_master:
                self.flat_param = nn.Parameter(torch.zeros(padded_size, device='cuda'))
                # if self.is_master: self._measure_gpu_memory(f"After creating flat_param and grad ")
                self.update_module_params(include_grads=False)
                for m in self.module_list.modules():
                    if len(list(m.children())) == 0 and hasattr(m, 'reset_parameters'): 
                        m.reset_parameters() #doesn't break weight sharing scheme since it's an in place operation
                self.module_list.apply(param_init_fn)
                flat_param_shards = list(self.flat_param.chunk(self.world_size))

            dist.scatter(self.local_shard.data, flat_param_shards if self.is_master else None, src=0)
        
        self.flat_param = nn.Parameter(self.local_shard)
        self.flat_param.grad = self.local_shard.grad
        self.update_module_params(include_grads=False)


    def update_module_params(self, include_grads, flag=False):
        is_sharded = self.flat_param.data_ptr() == self.local_shard.data_ptr()
        local_shard_size = self.local_shard.numel()
        offset = 0 - local_shard_size * self.rank if is_sharded else 0
        for name, numel, shape in zip(self.param_names, self.param_numels, self.param_shapes):
            if is_sharded:
                # print(name, offset, offset + numel, 0, local_shard_size, offset >= local_shard_size, offset + numel < 0, self.rank)
                if offset >= local_shard_size or offset + numel < 0:
                    param_tensor = torch.empty(0, device='cuda')
                    grad_tensor = None
                else:
                    start = max(offset, 0)
                    end = min(offset+numel, local_shard_size)
                    param_tensor = self.local_shard[start:end]
                    grad_tensor = self.local_shard.grad[start:end] if include_grads else None
            else:
                param_tensor = self.flat_param[offset:offset+numel].view(shape)
                grad_tensor = self.flat_param.grad[offset:offset+numel].view(shape) if include_grads else None
            name_parts = name.split('.')
            module = self.module_list #clean up naming
            for part in name_parts[:-1]:
                module = getattr(module, part)
            if getattr(module, name_parts[-1]).device.type == 'meta':
                setattr(module, name_parts[-1], nn.Parameter(param_tensor))
            else:
                getattr(module, name_parts[-1]).data = param_tensor

            if include_grads:
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
  
    def gather(self, include_grads=False, flag=True):
        params_gathered = self.flat_param.data_ptr() != self.local_shard.data_ptr()
        if not params_gathered:
            # if flag==True and self.is_master: print(f"Gather for backward through fsdp unit: {self.unit_name}")
            # if self.is_master: self._measure_gpu_memory(f"Before gathering unit {self.unit_name}")
            full_tensor = torch.zeros(self.local_shard.numel() * self.world_size, device=self.local_shard.device)
            # if self.is_master: self._measure_gpu_memory(f"After creating full tensor for {self.unit_name}")
            dist.all_gather_into_tensor(full_tensor, self.local_shard)
            self.flat_param.data = full_tensor

        if include_grads:
            full_grads_tensor = torch.zeros(self.local_shard.grad.numel() * self.world_size, device=self.local_shard.grad.device)
            self.flat_param.grad = full_grads_tensor

        self.update_module_params(include_grads=include_grads)

            # if flag:
            #     if not self.is_master: print("before")

            # if self.is_master: 
            #     self._measure_gpu_memory(f"After gathering unit {self.unit_name}", after_flag=True)
            #     print(f"Increase: {self.after_buffer - self.before_buffer}")

    def pre_backward(self, include_grads, flag):
        self.grad_counter = 0
        self.gather(include_grads=include_grads, flag=flag)        

    def post_backward(self, name, flag=False):
        self.grad_counter += 1
        # if not self.is_master: print(self.grad_counter, len(list(self.module_list.parameters())), name)
        if self.grad_counter == len(list(self.module_list.parameters())):
            grad_shards = list(self.flat_param.grad.chunk(self.world_size))
            buffer = torch.empty(self.local_shard.shape, device='cuda')
            dist.reduce_scatter(buffer, grad_shards, op=dist.ReduceOp.AVG)
            self.local_shard.grad.add_(buffer)
            self.shard(include_grads=True, flag=True)
            # if self.is_master: print(f"After sharding")

    def shard(self, include_grads=False, flag=False):
        if self.flat_param.data_ptr() != self.local_shard.data_ptr(): #check if flat_params is not sharded            
            # if self.is_master: self._measure_gpu_memory(f"Before sharding unit {self.unit_name}", before_flag=True)
            self.flat_param.data = self.local_shard.data
            if include_grads:
                self.flat_param.grad = self.local_shard.grad

            self.update_module_params(include_grads=include_grads)
            
            # if flag==True and self.is_master: print(f"Shard after backward through fsdp unit: {self.unit_name}")

            # if self.is_master: 
            #     self._measure_gpu_memory(f"After sharding unit {self.unit_name}", after_flag=True)
            #     print(f"Decrease: {self.before_buffer - self.after_buffer}")
    
class CustomFSDP(torch.nn.Module):
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
            gpt_model.transformer.wte, 
            gpt_model.transformer.wpe, 
            gpt_model.transformer.ln_f, 
            gpt_model.lm_head
        ])
        fsdp_units.append(FSDPUnit(remaining_params, param_init_fn, self.world_size, self.rank, "remaining"))
        return fsdp_units
        
    def _register_hooks(self):
        for i, unit in enumerate(self.fsdp_units):
            for j, module in enumerate(unit.module_list):
                if j == 0 and len(unit.module_list) == 1:
                    module.register_forward_pre_hook(lambda m, i, u=unit: u.gather()) 
                elif j==1 and len(unit.module_list) != 1:
                    module.register_forward_pre_hook(lambda m, i, u=unit: u.gather()) 
                # Only add post-forward hook to the last module in each unit except for the last fsdp unit 
                # (wrt declared order, not logical order) 
                # CHANGE
                if j == len(unit.module_list)-1: 
                    if len(unit.module_list) == 1: 
                        module.register_forward_hook(lambda m, i, o, u=unit: u.shard()) 
                    module.register_full_backward_pre_hook(lambda m, go, u=unit: u.pre_backward(include_grads=True, flag=False))

            for name, param in unit.module_list.named_parameters():
                param.register_post_accumulate_grad_hook(lambda p, u=unit, n=name: u.post_backward(n))

    @torch.no_grad()
    def clip_grad_norm(self, max_norm):
        local_grad_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(p.grad, 2.0, dtype=torch.float32) 
                    for p in self.module.parameters()
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
        output = self.module(*args, **kwargs)
        return output