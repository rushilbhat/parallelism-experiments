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
    def forward(ctx, tensor, op, group):
        ctx.op = op
        if op == 'sum':
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        # elif op == 'max':
        #     pre_reduce = tensor.clone()
        #     dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
        #     mask = (pre_reduce == tensor).float()
        #     count_mask = mask.clone()
        #     dist.all_reduce(count_mask, op=dist.ReduceOp.SUM, group=group)
        #     ctx.save_for_backward(mask, count_mask)
        else:
            raise ValueError(f"Unsupported reduce op. Use dist.ReduceOp.SUM or dist.ReduceOp.MAX.")
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.op == 'sum':
            return grad_output, None, None
        # elif ctx.op == 'max':
        #     mask, count_mask = ctx.saved_tensors
        #     grad_input = grad_output * mask / count_mask
        #     return grad_input, None, None


class ColParallelLinear(nn.Module):
    def __init__(self, linear, gather_output, tp_group):
        super().__init__()
        self.linear = linear
        self.tp_group = tp_group
        self.gather_output = gather_output
        self.tp_world_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)

        local_out_features = self.linear.out_features // self.tp_world_size
        in_features = self.linear.in_features

        sharded_weight = nn.Parameter(torch.empty(local_out_features, 
                                                  in_features, 
                                                  device=self.linear.weight.device, 
                                                  dtype=self.linear.weight.dtype
                                                  ))
        
        start_idx = self.tp_rank * local_out_features
        end_idx = start_idx + local_out_features

        if self.linear.weight.device.type != 'meta':
            sharded_weight.data.copy_(self.linear.weight.data[start_idx:end_idx])
        self.linear.weight = sharded_weight

        if self.linear.bias is not None:
            sharded_bias = nn.Parameter(torch.empty(local_out_features, 
                                                    device=self.linear.bias.device, 
                                                    dtype=self.linear.bias.dtype))
            if self.linear.bias.device.type != 'meta':
                sharded_bias.data.copy_(self.linear.bias.data[start_idx:end_idx])
            self.linear.bias = sharded_bias

    def forward(self, x: torch.Tensor):
        if x._backward_hooks is None:
            x.register_hook(lambda grad: dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.tp_group))

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
        self.tp_rank = dist.get_rank(tp_group)

        local_in_features = self.linear.in_features // self.tp_world_size
        out_features = self.linear.out_features

        sharded_weight = nn.Parameter(torch.empty(out_features, 
                                                  local_in_features, 
                                                  device=self.linear.weight.device, 
                                                  dtype=self.linear.weight.dtype
                                                  ))

        start_idx = self.tp_rank * local_in_features
        end_idx = start_idx + local_in_features

        if self.linear.weight.device.type != 'meta':
            sharded_weight.data.copy_(self.linear.weight.data[:, start_idx:end_idx])
        self.linear.weight = sharded_weight

        if self.linear.bias is not None:
            sharded_bias = nn.Parameter(torch.empty(out_features, 
                                                    device=self.linear.bias.device, 
                                                    dtype=self.linear.bias.dtype))
            if self.linear.bias.device.type != 'meta':
                sharded_bias.data.copy_(self.linear.bias.data)
            self.linear.bias = sharded_bias

    def forward(self, x: torch.Tensor):
        local_output = self.linear(x)

        if self.reduce_output:
            local_output = DifferentiableAllReduce.apply(local_output, 'sum', self.tp_group)

        return local_output

class VocabParallelEmbedding(nn.Module):
    def __init__(self, embedding, tp_group):
        super().__init__()
        self.embedding = embedding
        self.tp_group = tp_group
        self.tp_world_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)

        self.local_num_embeddings = embedding.num_embeddings // self.tp_world_size
        self.vocab_start_idx = self.tp_rank * self.local_num_embeddings
        self.vocab_end_idx = self.vocab_start_idx + self.local_num_embeddings

        embedding_dim = embedding.embedding_dim

        sharded_weight = nn.Parameter(torch.empty(self.local_num_embeddings, 
                                                  embedding_dim,
                                                  device=embedding.weight.device,
                                                  dtype=embedding.weight.dtype))
        
        if self.embedding.weight.device.type != 'meta':
            sharded_weight.data.copy_(self.embedding.weight.data[self.vocab_start_idx:self.vocab_end_idx])
        self.embedding.weight = sharded_weight

        
    def forward(self, input_ids):
        local_mask = (input_ids >= self.vocab_start_idx) & (input_ids < self.vocab_end_idx)    
        local_ids = input_ids - self.vocab_start_idx
        clamped_local_ids = torch.clamp(local_ids, 0, self.local_num_embeddings - 1)
        local_output = self.embedding(clamped_local_ids)
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

    # local_logits_max, _ = logits.max(dim=-1, keepdim=True)
    # global_logits_max = DifferentiableAllReduce.apply(local_logits_max, 'max', tp_group)
    # shifted_logits = logits - global_logits_max
    # No need to backpropagating through logits_max since the subtracting the max is a constant and will not affect the gradient
    logits_max = logits.detach().max(dim=-1, keepdim=True)[0]        
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
    shifted_logits = logits - logits_max
    exp_logits = torch.exp(shifted_logits)
    
    local_sum = exp_logits.sum(dim=-1, keepdim=True)
    global_sum = DifferentiableAllReduce.apply(local_sum, 'sum', tp_group)
    logsum = torch.log(global_sum)
    log_probs = shifted_logits - logsum

    ignore_mask = (targets != ignore_index)
    local_mask = (targets >= vocab_start_idx) & (targets < vocab_end_idx) & ignore_mask

    local_targets = targets - vocab_start_idx
    local_targets = local_targets.clamp(0, local_vocab_size - 1)

    local_pred_log_probs = torch.gather(log_probs, dim=-1, index=local_targets.unsqueeze(-1)).squeeze(-1)
    local_pred_log_probs = local_pred_log_probs * local_mask.float()

    local_loss = local_pred_log_probs.sum()
    global_loss = DifferentiableAllReduce.apply(local_loss, 'sum', tp_group)
    valid_count = ignore_mask.float().sum()
    avg_nll_loss = -global_loss / valid_count

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
                        new_child = RowParallelLinear(module, shard_info.get('reduce_output', False), tp_group)
                    elif shard_info['type'] == 'col':
                        new_child = ColParallelLinear(module, shard_info.get('gather_output', False), tp_group)
                    elif shard_info['type'] == 'vocab':
                        new_child = VocabParallelEmbedding(module, tp_group)
                    else:
                        raise ValueError(f"Invalid shard type {shard_info['type']}")
                    
                    *parent_path, leaf = module_name.split('.')
                    parent_module = reduce(getattr, parent_path, model)
                    setattr(parent_module, leaf, new_child)
                    break

    if config.tie_word_embeddings:
        model.transformer.wte.embedding.weight = model.lm_head.linear.weight