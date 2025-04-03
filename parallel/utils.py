#parallel/utils.py
import torch
import torch.distributed as dist
from .tensor_parallel import RowParallelLinear, ColParallelLinear, VocabParallelEmbedding

@torch.no_grad()
def clip_grad_norm(model, max_norm, tp_group, dp_group, dp_type):
    if dist.get_world_size(tp_group) > 1:
        tp_sharded_params  = []
        tp_replicated_params = []
        def separate_tp_sharded_params(module):
            for name, child in module.named_children():
                if isinstance(child, (RowParallelLinear)):
                    tp_sharded_params.append(child.weight)
                    if child.bias is not None:
                        tp_replicated_params.append(child.bias)
                elif isinstance(child, (ColParallelLinear)):
                    if not (model.config.tie_word_embeddings and child == model.lm_head):
                        tp_sharded_params.append(child.weight)
                        if child.bias is not None:
                            tp_sharded_params.append(child.bias)
                elif isinstance(child, (VocabParallelEmbedding)):
                    tp_sharded_params.append(child.weight)
                else:                                                                         
                    for n, p in child.named_parameters(recurse=False):
                        if not (model.config.tie_word_embeddings and child == model.lm_head):
                            tp_replicated_params.append(p)
                    separate_tp_sharded_params(child)

        separate_tp_sharded_params(model)


        tp_sharded_squared_grad_norm = sum(p.grad.pow(2).sum() for p in tp_sharded_params if p.grad is not None)
        dist.all_reduce(tp_sharded_squared_grad_norm, group=tp_group)
        tp_replicated_squared_grad_norm = sum(p.grad.pow(2).sum() for p in tp_replicated_params if p.grad is not None)
        total_squared_grad_norm = tp_sharded_squared_grad_norm + tp_replicated_squared_grad_norm

        if dist.get_world_size(dp_group) > 1 and dp_type == "fsdp":
            dist.all_reduce(total_squared_grad_norm, group=dp_group)

    else:
        total_squared_grad_norm = sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None)
        if dist.get_world_size(dp_group) > 1 and dp_type == "fsdp":
            dist.all_reduce(total_squared_grad_norm, group=dp_group)
        
    grad_norm = total_squared_grad_norm ** 0.5
    clip_coef = max_norm / (grad_norm + 1e-6)
    if clip_coef < 1.0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(clip_coef)

    return grad_norm