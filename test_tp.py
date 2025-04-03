import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import unittest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tensor_parallel import ColParallelLinear, RowParallelLinear, VocabParallelEmbedding, apply_tensor_parallelism, DifferentiableAllGather, DifferentiableAllReduce, vocab_parallel_cross_entropy_loss
from model import GPTConfig, GPT
from test_distributed_base import BaseDistributedTest
torch.use_deterministic_algorithms(True)

class TestTP(BaseDistributedTest):
    IN_SIZE = 6
    OUT_SIZE = 12
    BIAS = True
    DTYPE = torch.float32
    SEED = 42
    BATCH_SIZE = 4

    def test_col_parallel_linear_sharding_consistency(self):
        torch.manual_seed(self.SEED)
        with torch.device(self.device):
            unsharded = nn.Linear(self.IN_SIZE, self.OUT_SIZE, bias=self.BIAS, dtype=self.DTYPE)

        col_parallel = ColParallelLinear(
            in_features=self.IN_SIZE,
            out_features=self.OUT_SIZE,
            bias=self.BIAS,
            device=self.device,
            dtype=self.DTYPE,
            gather_output=False,
            tp_group=dist.group.WORLD
        )
        col_parallel.copy(unsharded)

        expected_weight = torch.chunk(unsharded.weight, self.world_size, dim=0)[self.rank]
        expected_bias = torch.chunk(unsharded.bias, self.world_size, dim=0)[self.rank]

        self.assertEqual(col_parallel.weight.shape, expected_weight.shape)
        self.assertEqual(col_parallel.bias.shape, expected_bias.shape)
        self.assertEqual(col_parallel.weight.device, torch.device(self.device))
        self.assertEqual(col_parallel.bias.device, torch.device(self.device))
        self.assertTrue(torch.equal(col_parallel.weight, expected_weight))
        self.assertTrue(torch.equal(col_parallel.bias, expected_bias))

    def test_col_parallel_linear_forward_backward_numerical_correctness(self):
        torch.manual_seed(self.SEED)
        with torch.device(self.device):
            unsharded = nn.Linear(self.IN_SIZE, self.OUT_SIZE, bias=self.BIAS, dtype=self.DTYPE)

        col_parallel = ColParallelLinear(
            in_features=self.IN_SIZE,
            out_features=self.OUT_SIZE,
            bias=self.BIAS,
            device=self.device,
            dtype=self.DTYPE,
            gather_output=True,
            tp_group=dist.group.WORLD
        )
        col_parallel.copy(unsharded)
        
        ref_input = torch.randn(self.BATCH_SIZE, self.IN_SIZE, device=self.device, requires_grad=True)
        expected_output = unsharded(ref_input)
        
        test_input = ref_input.detach().clone().requires_grad_(True)
        output_tensor = col_parallel(test_input)

        self.assertTrue(torch.equal(output_tensor, expected_output))

        random_tensor = torch.randn_like(expected_output)
        ref_loss = (expected_output * random_tensor).sum()
        ref_loss.backward()
        test_loss = (output_tensor * random_tensor).sum()
        test_loss.backward()
        
        self.assertTrue(torch.allclose(torch.chunk(unsharded.weight.grad, self.world_size, dim=0)[self.rank], col_parallel.weight.grad))
        self.assertTrue(torch.allclose(torch.chunk(unsharded.bias.grad, self.world_size, dim=0)[self.rank], col_parallel.bias.grad))
        self.assertTrue(torch.allclose(ref_input.grad, test_input.grad))

    def test_row_parallel_linear_sharding_consistency(self):
        torch.manual_seed(self.SEED)
        with torch.device(self.device):
            unsharded = nn.Linear(self.IN_SIZE, self.OUT_SIZE, bias=self.BIAS, dtype=self.DTYPE)

        row_parallel = RowParallelLinear(
            in_features=self.IN_SIZE,
            out_features=self.OUT_SIZE,
            bias=self.BIAS,
            device=self.device,
            dtype=self.DTYPE,
            reduce_output=False,
            tp_group=dist.group.WORLD
        )
        row_parallel.copy(unsharded)

        expected_weight = torch.chunk(unsharded.weight, self.world_size, dim=1)[self.rank]
        expected_bias = unsharded.bias

        self.assertEqual(row_parallel.weight.shape, expected_weight.shape)
        self.assertEqual(row_parallel.bias.shape, expected_bias.shape)
        self.assertEqual(row_parallel.weight.device, torch.device(self.device))
        self.assertEqual(row_parallel.bias.device, torch.device(self.device))
        self.assertTrue(torch.equal(row_parallel.weight, expected_weight))
        self.assertTrue(torch.equal(row_parallel.bias, expected_bias))

    def test_row_parallel_linear_forward_backward_numerical_correctness(self):
        torch.manual_seed(self.SEED)
        with torch.device(self.device):
            unsharded = nn.Linear(self.IN_SIZE, self.OUT_SIZE, bias=self.BIAS, dtype=self.DTYPE)

        row_parallel = RowParallelLinear(
            in_features=self.IN_SIZE,
            out_features=self.OUT_SIZE,
            bias=self.BIAS,
            device=self.device,
            dtype=self.DTYPE,
            reduce_output=True,
            tp_group=dist.group.WORLD
        )
        row_parallel.copy(unsharded)

        ref_input = torch.randn(self.BATCH_SIZE, self.IN_SIZE, device=self.device, requires_grad=True)
        expected_output = unsharded(ref_input)

        test_input = ref_input.detach().clone().requires_grad_(True)
        local_test_input = torch.chunk(test_input, self.world_size, dim=-1)[self.rank]
        local_test_input.retain_grad()
        output_tensor = row_parallel(local_test_input)

        self.assertTrue(torch.allclose(output_tensor, expected_output))

        random_tensor = torch.randn_like(expected_output)
        ref_loss = (expected_output * random_tensor).sum()
        ref_loss.backward()
        test_loss = (output_tensor * random_tensor).sum()
        test_loss.backward()
        
        self.assertTrue(torch.allclose(torch.chunk(unsharded.weight.grad, self.world_size, dim=-1)[self.rank], row_parallel.weight.grad))
        self.assertTrue(torch.allclose(unsharded.bias.grad, row_parallel.bias.grad))
        self.assertTrue(torch.allclose(torch.chunk(ref_input.grad, self.world_size, dim=-1)[self.rank], local_test_input.grad))
        

    def test_vocab_parallel_embedding_sharding_consistency(self):
        torch.manual_seed(self.SEED)
        with torch.device(self.device):
            unsharded = nn.Embedding(self.IN_SIZE, self.OUT_SIZE, device=self.device, dtype=self.DTYPE)

        vocab_parallel = VocabParallelEmbedding(
            num_embeddings=self.IN_SIZE,
            embedding_dim=self.OUT_SIZE,
            device=self.device,
            dtype=self.DTYPE,
            tp_group=dist.group.WORLD
        )
        vocab_parallel.copy(unsharded)

        expected_weight = torch.chunk(unsharded.weight, self.world_size, dim=0)[self.rank]

        self.assertEqual(vocab_parallel.weight.shape, expected_weight.shape)
        self.assertEqual(vocab_parallel.weight.device, torch.device(self.device))
        self.assertTrue(torch.equal(vocab_parallel.weight, expected_weight))


    def test_vocab_parallel_embedding_forward_backward_numerical_correctness(self):
        torch.manual_seed(self.SEED)
        with torch.device(self.device):
            unsharded = nn.Embedding(self.IN_SIZE, self.OUT_SIZE, device=self.device, dtype=self.DTYPE)

        vocab_parallel = VocabParallelEmbedding(
            num_embeddings=self.IN_SIZE,
            embedding_dim=self.OUT_SIZE,
            device=self.device,
            dtype=self.DTYPE,
            tp_group=dist.group.WORLD
        )
        vocab_parallel.copy(unsharded)

        ref_input = torch.randint(0, self.IN_SIZE, (self.BATCH_SIZE,), device=self.device)
        expected_output = unsharded(ref_input)

        test_input = ref_input.detach().clone()
        output_tensor = vocab_parallel(test_input)

        self.assertTrue(torch.equal(output_tensor, expected_output))

        random_tensor = torch.randn_like(expected_output)
        ref_loss = (expected_output * random_tensor).sum()
        ref_loss.backward()
        test_loss = (output_tensor * random_tensor).sum()
        test_loss.backward()

        self.assertTrue(torch.equal(torch.chunk(unsharded.weight.grad, self.world_size, dim=0)[self.rank], vocab_parallel.weight.grad))
    
    def test_vocab_parallel_cross_entropy_loss_forward_backward_numerical_correctness(self):        
        seq_len = self.IN_SIZE
        vocab_size = self.OUT_SIZE     

        ref_logits = torch.randn(self.BATCH_SIZE, seq_len, vocab_size, device=self.device, dtype=self.DTYPE, requires_grad=True)    
        targets = torch.randint(0, vocab_size, (self.BATCH_SIZE, seq_len), device=self.device)
        reference_loss = F.cross_entropy(ref_logits.reshape(-1, vocab_size), targets.reshape(-1))

        test_logits = ref_logits.detach().clone().requires_grad_(True)
        local_logits = torch.chunk(test_logits, self.world_size, dim=-1)[self.rank]
        local_logits.retain_grad()
        test_loss = vocab_parallel_cross_entropy_loss(local_logits, targets, tp_group=dist.group.WORLD)
        
        self.assertTrue(torch.allclose(reference_loss, test_loss))

        reference_loss.backward()
        test_loss.backward()

        self.assertTrue(torch.allclose(torch.chunk(ref_logits.grad, self.world_size, dim=-1)[self.rank], local_logits.grad))

    def test_memory_reduction_gpt(self):
        """
        Check that tensor parallelism reduces peak GPU memory usage compared to unsharded model.
        """
        with torch.device(self.device):
            model = self.create_gpt_model(n_layer=12)

        pre_shard_alloc = torch.cuda.memory_allocated(self.device) / 2**20
        pre_shard_estimate = sum(p.numel()*p.element_size() for p in model.parameters()) / 2**20

        apply_tensor_parallelism(model, dist.group.WORLD, enable_loss_parallelism=False)

        post_shard_alloc = torch.cuda.memory_allocated(self.device) / 2**20
        post_shard_estimate = sum(p.numel()*p.element_size() for p in model.parameters()) / 2**20
        
        actual_reduction = (post_shard_alloc - pre_shard_alloc) / pre_shard_alloc
        estimated_reduction = (post_shard_estimate - pre_shard_estimate) / pre_shard_estimate

        self.assertAlmostEqual(actual_reduction, estimated_reduction, delta=0.1)

    def test_differentiable_all_gather_forward_backward_numerical_correctness(self):
        local_t = torch.randn(self.BATCH_SIZE, self.IN_SIZE, device=self.device, requires_grad=True)

        ref_list = [torch.empty_like(local_t) for _ in range(self.world_size)]
        dist.all_gather(ref_list, local_t.detach(), group=dist.group.WORLD)
        ref_gather = torch.cat(ref_list, dim=-1)

        gathered = DifferentiableAllGather.apply(local_t, dist.group.WORLD)

        self.assertTrue(torch.equal(ref_gather, gathered))

        loss = gathered.sum()
        loss.backward()

        expected_grad = torch.ones_like(local_t)
        self.assertTrue(torch.equal(local_t.grad, expected_grad))

    def test_differentiable_all_reduce_forward_backward_numerical_correctness(self):
        local_t = torch.randn(self.BATCH_SIZE, self.IN_SIZE, device=self.device, requires_grad=True)

        ref = local_t.detach().clone()
        dist.all_reduce(ref, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

        reduced = DifferentiableAllReduce.apply(local_t, 'sum', dist.group.WORLD)

        self.assertTrue(torch.equal(ref, reduced))

        loss = reduced.sum()
        loss.backward()

        expected_grad = torch.ones_like(local_t)
        self.assertTrue(torch.equal(local_t.grad, expected_grad))


if __name__ == "__main__":
    unittest.main()

