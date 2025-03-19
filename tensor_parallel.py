import torch
import torch.nn as nn
import torch.distributed as dist
from functools import reduce

class DifferentiableAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        world_size = dist.get_world_size(group)
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gather_list, tensor, group=group)

        return torch.cat(gather_list, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        world_size = dist.get_world_size(group)
        local_chunk = torch.chunk(grad_output, world_size, dim=-1)[dist.get_rank(group)]
        return local_chunk, None
    
class DifferentiableAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class ColParallelLinear(nn.Module):
    def __init__(self, linear, gather_output, tp_group):
        super().__init__()
        self.linear = linear
        self.tp_group = tp_group
        self.gather_output = gather_output
        self.tp_world_size = dist.get_world_size(tp_group)

        local_out_features = self.linear.out_features // self.tp_world_size
        in_features = self.linear.in_features
        device = self.linear.weight.device
        dtype = self.linear.weight.dtype

        self.linear.weight = nn.Parameter(torch.empty(local_out_features, in_features, device=device, dtype=dtype))
        if self.linear.bias is not None:
            self.linear.bias = nn.Parameter(torch.empty(local_out_features, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        local_output = self.linear(x)

        if self.gather_output:
            out = DifferentiableAllGather.apply(local_output, self.tp_group)
        else:
            out = local_output

        return out


class RowParallelLinear(nn.Module):
    def __init__(self, linear, reduce_output, tp_group):
        super().__init__()
        self.linear = linear
        self.tp_group = tp_group
        self.reduce_output = reduce_output
        self.tp_world_size = dist.get_world_size(tp_group)

        local_in_features = self.linear.in_features // self.tp_world_size
        out_features = self.linear.out_features
        device = self.linear.weight.device
        dtype = self.linear.weight.dtype

        self.linear.weight = nn.Parameter(torch.empty(out_features, local_in_features, device=device, dtype=dtype))
        if self.linear.bias is not None:
            self.linear.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        local_output = self.linear(x)

        if self.reduce_output:
            local_output = DifferentiableAllReduce.apply(local_output, self.tp_group)

        return local_output

def apply_tensor_parallelism(model: nn.Module,
                            sharding_config: dict,
                            tp_group,
                            reduce_row_output=False,
                            gather_col_output=False):
    for fname, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            for linear_layer_name, split_dim in sharding_config.items():
                if fname.endswith(linear_layer_name):
                    if split_dim == 0:
                        new_child = RowParallelLinear(module, reduce_row_output, tp_group)
                    elif split_dim == 1:
                        new_child = ColParallelLinear(module, gather_col_output, tp_group)
                    else:
                        raise ValueError(f"Invalid shard dimension {split_dim} for {module}")
                    
                    *parent_path, leaf = fname.split('.')
                    parent_module = reduce(getattr, parent_path, model)
                    setattr(parent_module, leaf, new_child)
                    break