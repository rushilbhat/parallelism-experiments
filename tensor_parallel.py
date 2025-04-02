import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def forward(ctx, tensor, op, group):
        ctx.op = op
        if op == 'sum':
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        else:
            raise ValueError(f"Unsupported reduce op. Use dist.ReduceOp.SUM or dist.ReduceOp.MAX.")
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.op == 'sum':
            return grad_output, None, None


class ColParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias, device, dtype, gather_output, tp_group):
        self.tp_group = tp_group
        self.tp_world_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)
        self.gather_output = gather_output

        self.local_out_features = out_features // self.tp_world_size
        super().__init__(in_features, self.local_out_features, bias=bias, device=device, dtype=dtype)
    
    def copy(self, unsharded_mod: nn.Linear):
        start_idx = self.tp_rank * self.local_out_features
        end_idx = start_idx + self.local_out_features
        self.weight.data.copy_(unsharded_mod.weight.data[start_idx:end_idx])
        if self.bias is not None:
            self.bias.data.copy_(unsharded_mod.bias.data[start_idx:end_idx])

    def forward(self, x: torch.Tensor):
        if x._backward_hooks is None:
            x.register_hook(lambda grad: dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.tp_group))

        local_output = F.linear(x, self.weight, self.bias)

        if self.gather_output:
            out = DifferentiableAllGather.apply(local_output, self.tp_group)
        else:
            out = local_output

        return out


class RowParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias, device, dtype, reduce_output, tp_group):
        self.tp_group = tp_group
        self.tp_world_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)
        self.reduce_output = reduce_output

        self.local_in_features = in_features // self.tp_world_size
        super().__init__(self.local_in_features, out_features, bias=bias, device=device, dtype=dtype)

    def copy(self, unsharded_mod: nn.Linear):
        start_idx = self.tp_rank * self.local_in_features
        end_idx = start_idx + self.local_in_features
        self.weight.data.copy_(unsharded_mod.weight.data[:, start_idx:end_idx])
        if self.bias is not None:
            self.bias.data.copy_(unsharded_mod.bias.data)

    def forward(self, x: torch.Tensor):
        local_output = F.linear(x, self.weight)

        if self.reduce_output:
            local_output = DifferentiableAllReduce.apply(local_output, 'sum', self.tp_group)

        if self.bias is not None:
            local_output = local_output + self.bias

        return local_output

class VocabParallelEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, device, dtype, tp_group):
        self.tp_group = tp_group
        self.tp_world_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)

        self.local_num_embeddings = num_embeddings // self.tp_world_size
        super().__init__(self.local_num_embeddings, embedding_dim, device=device, dtype=dtype)
        
        self.vocab_start_idx = self.tp_rank * self.local_num_embeddings
        self.vocab_end_idx = self.vocab_start_idx + self.local_num_embeddings

    def copy(self, unsharded_mod: nn.Embedding):
        self.weight.data.copy_(unsharded_mod.weight.data[self.vocab_start_idx:self.vocab_end_idx])

    def forward(self, input_ids):
        local_mask = (input_ids >= self.vocab_start_idx) & (input_ids < self.vocab_end_idx)    
        local_ids = input_ids - self.vocab_start_idx
        clamped_local_ids = torch.clamp(local_ids, 0, self.local_num_embeddings - 1)
        local_output = super().forward(clamped_local_ids)
        local_output = local_output * local_mask.float().unsqueeze(-1)
        
        local_output = DifferentiableAllReduce.apply(local_output, 'sum', self.tp_group)

        return local_output

def vocab_parallel_cross_entropy_loss(logits, targets, tp_group, ignore_index=-100):
    # local_logits: (B, T, V_local)
    # targets: (B, T) full target indices in [0, V)
    # vocab_start, vocab_end: the range of token indices on this rank

    tp_rank = dist.get_rank(tp_group)
    local_vocab_size = logits.size(-1)
    vocab_start_idx = tp_rank * local_vocab_size
    vocab_end_idx = vocab_start_idx + local_vocab_size

    exp_logits = torch.exp(logits)
    local_sum = exp_logits.sum(dim=-1)
    global_sum = DifferentiableAllReduce.apply(local_sum, 'sum', tp_group)
    logsumexp = torch.log(global_sum)

    local_mask = (targets >= vocab_start_idx) & (targets < vocab_end_idx) #& ignore_mask

    local_targets = targets - vocab_start_idx
    local_targets = local_targets.clamp(0, local_vocab_size - 1)

    pred_logits = torch.gather(logits, dim=-1, index=local_targets.unsqueeze(-1)).squeeze(-1)
    pred_logits = pred_logits * local_mask.float()
    pred_logits = DifferentiableAllReduce.apply(pred_logits, 'sum', tp_group)
    pred_logprobs = logsumexp - pred_logits
    
    avg_nll_loss = pred_logprobs.mean()

    return avg_nll_loss


def apply_tensor_parallelism(model: nn.Module,
                            tp_group,
                            enable_loss_parallelism):
    
    sharding_map = {
        'attn.c_attn1':  {'type': 'col', 'gather_output': False},
        'attn.c_attn2':  {'type': 'col', 'gather_output': False},
        'attn.c_attn3':  {'type': 'col', 'gather_output': False},
        'attn.c_proj':  {'type': 'row', 'reduce_output': True},
        'mlp.c_fc':     {'type': 'col', 'gather_output': False},
        'mlp.c_proj':   {'type': 'row', 'reduce_output': True},
        'wte':          {'type': 'vocab'},
        'lm_head':      {'type': 'col', 'gather_output': False if enable_loss_parallelism else True},
    }

    config = model.config
    tp_size = dist.get_world_size(tp_group)
    assert config.n_embd % tp_size == 0, f"n_embd={config.n_embd} is not divisible by TP size {tp_size}."
    assert config.n_head % tp_size == 0, f"n_head={config.n_head} is not divisible by TP size {tp_size}."
    assert config.vocab_size % tp_size == 0, f"vocab_size={config.vocab_size} is not divisible by TP size {tp_size}."

    for module_name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            for key, shard_info in sharding_map.items():
                if module_name.endswith(key):
                    if shard_info['type'] == 'row':
                        sharded_module = RowParallelLinear(
                            in_features=module.in_features,
                            out_features=module.out_features,
                            bias=(module.bias is not None),
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                            tp_group=tp_group,
                            reduce_output=shard_info.get("reduce_output", False),
                        )
                    elif shard_info['type'] == 'col':
                        sharded_module = ColParallelLinear(
                            in_features=module.in_features,
                            out_features=module.out_features,
                            bias=(module.bias is not None),
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                            tp_group=tp_group,
                            gather_output=shard_info.get("gather_output", False),
                        )
                    elif shard_info['type'] == 'vocab':
                        sharded_module = VocabParallelEmbedding(
                            num_embeddings=module.num_embeddings,
                            embedding_dim=module.embedding_dim,
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                            tp_group=tp_group,
                        )
                    else:
                        raise ValueError(f"Invalid shard type {shard_info['type']}")

                    if module.weight.device.type != "meta":
                        sharded_module.copy(module)

                    *parent_path, leaf = module_name.split('.')
                    parent_module = reduce(getattr, parent_path, model)
                    setattr(parent_module, leaf, sharded_module)
                    break

    if config.tie_word_embeddings:
        model.transformer.wte._parameters = model.lm_head._parameters