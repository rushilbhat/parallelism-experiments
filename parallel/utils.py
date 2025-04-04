#parallel/utils.py
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.distributed as dist
from .tensor_parallel import RowParallelLinear, ColParallelLinear, VocabParallelEmbedding

torch.use_deterministic_algorithms(True)

@torch.no_grad()
def clip_grad_norm(model, max_norm, tp_group, dp_group, dp_type):
    tp_sharded_params = {}
    non_tp_sharded_params = {}
    def separate_tp_sharded_params(module):
        for name, child in module.named_children():
            if isinstance(child, RowParallelLinear):
                tp_sharded_params[id(child.weight)] = child.weight
                if child.bias is not None:
                    non_tp_sharded_params[id(child.bias)] = child.bias
            elif isinstance(child, ColParallelLinear):
                tp_sharded_params[id(child.weight)] = child.weight
                if child.bias is not None:
                    tp_sharded_params[id(child.bias)] = child.bias
            elif isinstance(child, VocabParallelEmbedding):
                tp_sharded_params[id(child.weight)] = child.weight
            else:                         
                for n, p in child.named_parameters(recurse=False):
                    non_tp_sharded_params[id(p)] = p
                separate_tp_sharded_params(child)
    
    separate_tp_sharded_params(model)    

    tp_sharded_sq_grad_norm = sum(p.grad.pow(2).sum() for p in tp_sharded_params.values() if p.grad is not None)
    if dist.get_world_size(tp_group) > 1:
        dist.all_reduce(tp_sharded_sq_grad_norm, group=tp_group)
    
    non_tp_sharded_sq_grad_norm = sum(p.grad.pow(2).sum() for p in non_tp_sharded_params.values() if p.grad is not None)
    total_sq_grad_norm = non_tp_sharded_sq_grad_norm + tp_sharded_sq_grad_norm
    if dist.get_world_size(dp_group) > 1 and dp_type == "fsdp":
        dist.all_reduce(total_sq_grad_norm, group=dp_group)

    grad_norm = total_sq_grad_norm ** 0.5
    clip_coef = max_norm / (grad_norm + 1e-6)
    if clip_coef < 1.0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(clip_coef)

    return grad_norm