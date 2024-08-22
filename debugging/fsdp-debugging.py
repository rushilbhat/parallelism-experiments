import torch
import torch.nn as nn
import torch.distributed as dist
import math
import time
import gc
import sys

class FSDPUnit:
    def __init__(self, module_list, param_init_fn, world_size, rank, module_name):
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
        self.module_name=module_name
        self.theoretical_memory = sum(p.numel() * p.element_size() for p in self.module_list.parameters()) / (2 * 1024**2)
        

        self._record_param_metadata()
        self._create_and_shard_flat_param(param_init_fn)

        torch.cuda.empty_cache()
        gc.collect()

        if self.is_master: self._measure_gpu_memory(f"After sharding flat_param {self.module_name} ")

        self._register_hooks()


    def _measure_gpu_memory(self, stage):
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
        print(f"[Rank {self.rank}] {stage}:")
        print(f"  Allocated: {memory_allocated:.2f} MB")
        print(f"  Reserved:  {memory_reserved:.2f} MB")

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
        if self.is_master: self._measure_gpu_memory(f"Before creating shard {self.module_name}")
        self.shard = torch.empty(shard_size, device='cuda')
        if self.is_master: self._measure_gpu_memory(f"After creating shard {self.module_name}")
        
        with torch.no_grad():
            if self.is_master:
                self.flat_param = nn.Parameter(torch.zeros(padded_size, device='cuda'))
                if self.is_master: self._measure_gpu_memory(f"After creating flat_param {self.module_name}")
                self._assign_params()
                self.module_list.apply(param_init_fn)
                flat_param_shards = list(self.flat_param.chunk(self.world_size))

            dist.scatter(self.shard, flat_param_shards if self.is_master else None, src=0)
        
        self.flat_param = nn.Parameter(self.shard)
        self._assign_params()


    def _assign_params(self, flag=False):
        is_sharded = self.flat_param.data_ptr() == self.shard.data_ptr()
        offset = 0
        for name, numel, shape in zip(self.param_names, self.param_numels, self.param_shapes):
            if is_sharded:
                param_tensor = torch.empty(0, device='cuda')
            else:
                param_tensor = self.flat_param[offset:offset+numel].view(shape)
            name_parts = name.split('.')
            module = self.module_list #clean up naming
            for part in name_parts[:-1]:
                module = getattr(module, part)
            param = getattr(module, name_parts[-1])
            if param.device.type == 'meta':
                setattr(module, name_parts[-1], nn.Parameter(param_tensor))
            else:
                param.data = param_tensor
            offset += numel

        unique_params = dict(self.module_list.named_parameters())
        if len(self.shared_params) > 0:
            for duplicate, original in self.shared_params.items():
                name_parts = duplicate.split('.')
                module = self.module_list
                for part in name_parts[:-1]:
                    module = getattr(module, part)
                setattr(module, name_parts[-1], unique_params[original])



    def _register_hooks(self):
        def pre_forward_hook(module, input):
            if self.flat_param.data_ptr() == self.shard.data_ptr(): #check if all shards have been gathered
                if self.is_master: self._measure_gpu_memory(f"Before gathering unit {self.module_name}")
                full_tensor = torch.empty(self.shard.numel() * self.world_size, device=self.shard.device)
                if self.is_master: self._measure_gpu_memory(f"After creating full tensor for {self.module_name}")
                dist.all_gather_into_tensor(full_tensor, self.shard)
                self.flat_param.data = full_tensor
                
                self._assign_params(flag=True)
                if self.is_master: self._measure_gpu_memory(f"After gathering unit {self.module_name}")

        def post_forward_hook(module, input, output):
            if self.flat_param.data_ptr() != self.shard.data_ptr(): #check if flat_params is not sharded
                if self.is_master: self._measure_gpu_memory(f"Before sharding unit {self.module_name}")

                self.flat_param.data = self.shard.data
                self._assign_params()

                torch.cuda.empty_cache()
                gc.collect()
                if self.is_master: self._measure_gpu_memory(f"After sharding unit {self.module_name}")


        for i, module in enumerate(self.module_list):
            module.register_forward_pre_hook(pre_forward_hook)
            # if i == len(self.module_list) - 1: # Only add post-forward hook to the last module (wrt declared order, not logical order) 
            module.register_forward_hook(post_forward_hook)

class CustomFSDP(torch.nn.Module):
    def __init__(self, module, param_init_fn, world_size, rank):
        super().__init__()
        self.module = module
        self.world_size = world_size
        self.rank = rank
        self.fsdp_units = self._create_fsdp_units_for_gpt(self.module, param_init_fn)

    def _create_fsdp_units_for_gpt(self, gpt_model, param_init_fn):
        fsdp_units = []
        # theoretical_running_sum = 0

        # if self.rank == 0: 
        #     print(f"Theoretical memory for entire model: {sum(p.numel() * p.element_size() for p in self.module.parameters()) / (2 * 1024**2)}")


        # fsdp_units.append(FSDPUnit(self.module, param_init_fn, self.world_size, self.rank, "model"))
        # theoretical_running_sum += fsdp_units[-1].theoretical_memory
        # if self.rank == 0: 
        #     print(f" Theoretical memory running sum: {theoretical_running_sum}")
        #     self._measure_gpu_memory("Actual memory running sum")

        # for block in gpt_model.transformer.h:
        #     fsdp_units.append(FSDPUnit(nn.ModuleList([block]), param_init_fn, self.world_size, self.rank, "block"))
        #     theoretical_running_sum += fsdp_units[-1].theoretical_memory
        #     if self.rank == 0: 
        #         print(f" Theoretical memory running sum: {theoretical_running_sum}")
        #         self._measure_gpu_memory("Actual memory running sum")

        # remaining_params = nn.ModuleList([
        #     gpt_model.transformer.wte, 
        #     gpt_model.transformer.wpe, 
        #     gpt_model.transformer.ln_f, 
        #     gpt_model.lm_head
        # ])
        # fsdp_units.append(FSDPUnit(remaining_params, param_init_fn, self.world_size, self.rank, "remaining"))
        # theoretical_running_sum += fsdp_units[-1].theoretical_memory
        # if self.rank == 0: 
        #     print(f" Theoretical memory running sum: {theoretical_running_sum}")
        #     self._measure_gpu_memory("Actual memory running sum")

        
        for name, module in list(self.module.named_modules()):
            if len(list(module.children())) == 0:
                #TURN OFF WEIGHT TYING SCHEME
                fsdp_units.append(FSDPUnit(nn.ModuleList([module]), param_init_fn, self.world_size, self.rank, name))
                # theoretical_running_sum += fsdp_units[-1].theoretical_memory
                # if self.rank == 0: 
                #     print(f" Theoretical memory running sum: {theoretical_running_sum}")
                #     self._measure_gpu_memory("Actual memory running sum")

        # if self.rank == 0: 
        #     self._measure_gpu_memory("After memory for entire model")


        return fsdp_units

    def _measure_gpu_memory(self, stage):
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
        print(f"[Rank {self.rank}] {stage}:")
        print(f"  Allocated: {memory_allocated:.2f} MB")
        print(f"  Reserved:  {memory_reserved:.2f} MB")

    def _measure_tensor_memory(self, tensor, name):
        memory_bytes = tensor.element_size() * tensor.numel()
        memory_mb = memory_bytes / 1024**2
        print(f"[Rank {self.rank}] {name} memory: {memory_mb:.2f} MB element-size = {tensor.element_size()} no. elements: {tensor.numel()} ")
    
    def get_tensor_addresses(self, tensor):
        start_address = tensor.data_ptr()
        end_address = start_address + tensor.nelement() * tensor.element_size() - 1
        return start_address, end_address    

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        return output

    def _check_remaining_references(self, flat_param):
        print("[Rank 0] Checking for remaining references:")
        ref_count = sys.getrefcount(flat_param.data)
        print(f" {ref_count} references")

