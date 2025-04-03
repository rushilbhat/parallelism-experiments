import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import unittest
from unittest.mock import patch
import torch
import torch.distributed as dist
import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from distributed import CustomFSDP
from model import GPT, GPTConfig, Block
from test_distributed_base import BaseDistributedTest

torch.use_deterministic_algorithms(True)

class TestFSDP(BaseDistributedTest):
    def wrap_with_custom_fsdp(self, model):
        return CustomFSDP(
            module=model,
            process_group=dist.group.WORLD,
            param_init_fn=model._init_weights
        )

    def check_memory_change(self, unit, operation, include_grads=False, tolerance=0.1):
        shard_size_bytes = unit.local_shard.numel() * unit.local_shard.element_size()
        grad_size_bytes = unit.local_shard.grad.numel() * unit.local_shard.element_size() if include_grads else 0
        total_bytes = (shard_size_bytes + grad_size_bytes) * unit.world_size
        if operation == 'gather':
            expected_change = total_bytes / (2**20)
        elif operation == 'shard':
            expected_change = -total_bytes / (2**20)

        pre_op_alloc = torch.cuda.memory_allocated()
        if operation == 'gather':
            unit._gather(include_grads=include_grads)
        elif operation == 'shard':
            unit._shard(include_grads=include_grads)
        post_op_alloc = torch.cuda.memory_allocated()
        actual_change = (post_op_alloc - pre_op_alloc) / (2**20)
        tolerance = abs(expected_change) * tolerance
        self.assertAlmostEqual(actual_change, expected_change, delta=tolerance)


    def test_fsdp_construction(self):
        """
        Test that CustomFSDP properly wraps submodules (model and Block) .
        """
        model = self.create_gpt_model(n_layer=3)
        fsdp_model = self.wrap_with_custom_fsdp(model)

        self.assertIsInstance(fsdp_model, CustomFSDP)
        
        for i, module in enumerate(fsdp_model.module.transformer.h):
            self.assertIsInstance(module, CustomFSDP, 
                                 f"Block {i} is not wrapped in CustomFSDP")
            self.assertIsInstance(module._fsdp_wrapped_module, Block,
                                 f"Block {i}'s wrapped module is not a Block")
        
        self.assertNotIsInstance(fsdp_model.module.transformer.wte, CustomFSDP,
                               "Word token embedding should not be wrapped in CustomFSDP")
        self.assertNotIsInstance(fsdp_model.module.transformer.wpe, CustomFSDP,
                               "Position embedding should not be wrapped in CustomFSDP")
        self.assertNotIsInstance(fsdp_model.module.transformer.ln_f, CustomFSDP,
                               "Final layer norm should not be wrapped in CustomFSDP")
    
    def test_fsdp_shard_parameter_correctness(self):
        """
        Test that parameters are sharded across the 2 ranks as expected.
        Compare each sharded parameter against a PyTorch's native FSDP.
        """
        model = self.create_gpt_model()
        custom_fsdp = self.wrap_with_custom_fsdp(model)

        gpt2_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Block,
            },
        )
        native_model = self.create_gpt_model()
        native_fsdp = FSDP(
            native_model,
            process_group=dist.group.WORLD,
            auto_wrap_policy=gpt2_auto_wrap_policy,
            use_orig_params=True,
            param_init_fn=native_model._init_weights
        )

        for (custom_name, custom_param), (native_name, native_param) in zip(
            custom_fsdp.named_parameters(), native_fsdp.named_parameters()
        ):
            self.assertEqual(custom_name, native_name)
            self.assertEqual(custom_param.shape, native_param.shape)
            self.assertEqual(custom_param.device, native_param.device)
            self.assertEqual(custom_param.dtype, native_param.dtype)
            self.assertEqual(custom_param.requires_grad, native_param.requires_grad)
            self.assertTrue(torch.equal(custom_param, native_param))
            self.assertTrue(custom_param.grad is None)
            self.assertTrue(native_param.grad is None)

    def test_fsdp_sharding_memory_usage(self):
        """
        Test that GPU allocated memory decreases by roughly the amount of memory occupied
        by fully constructed flat parameter.
        """

        model = self.create_gpt_model()
        fsdp_model = self.wrap_with_custom_fsdp(model)

        root_unit = fsdp_model
        block_unit = fsdp_model.module.transformer.h[0]
        root_unit._gather()
        block_unit._gather()

        self.check_memory_change(root_unit, 'shard')
        self.check_memory_change(block_unit, 'shard')

    
    def test_fsdp_gather_parameter_correctness(self):
        """
        Test that parameters are properly reconstructed from their sharded state.
        """
        model = self.create_gpt_model()

        original_params = {n: p.detach().clone() for n, p in model.named_parameters()}
            
        fsdp_model = self.wrap_with_custom_fsdp(model)
        
        root_unit = fsdp_model
        block_unit = fsdp_model.module.transformer.h[0]
        root_unit._gather()
        block_unit._gather()
        for (custom_name, custom_param), (original_name, original_param) in zip(fsdp_model.named_parameters(), original_params.items()):
            self.assertEqual(custom_name.replace('_fsdp_wrapped_module.', ''), original_name)
            self.assertTrue(torch.equal(custom_param, original_param))


    def test_fsdp_gather_memory_usage(self):
        """
        Test that GPU allocated memory increases by roughly the amount of memory occupied
        by fully constructed flat parameter.
        """
        model = self.create_gpt_model()
        fsdp_model = self.wrap_with_custom_fsdp(model)

        root_unit = fsdp_model
        block_unit = fsdp_model.module.transformer.h[0]

        self.check_memory_change(root_unit, 'gather')
        self.check_memory_change(block_unit, 'gather')


    def test_fsdp_gather_gradient_correctness(self):
        """
        Test that gathering with gradients culminates in the gradients on each parameter being zero.
        """
        model = self.create_gpt_model()
        fsdp_model = self.wrap_with_custom_fsdp(model)

        root_unit = fsdp_model
        block_unit = fsdp_model.module.transformer.h[0]
        root_unit._gather(include_grads=True)
        block_unit._gather(include_grads=True)

        for param in fsdp_model.parameters():
            self.assertTrue(torch.equal(param.grad, torch.zeros_like(param.grad)))

    def test_fsdp_gather_include_grads_memory_usage(self):
        """
        Test that gathering with gradients leads to an increase in memory usage by roughly the amount of memory occupied
        by fully constructed flat parameter and its gradient.
        """
        model = self.create_gpt_model()
        fsdp_model = self.wrap_with_custom_fsdp(model)

        root_unit = fsdp_model
        block_unit = fsdp_model.module.transformer.h[0]

        self.check_memory_change(root_unit, 'gather', include_grads=True)
        self.check_memory_change(block_unit, 'gather', include_grads=True)

    def test_fsdp_shard_include_grads_memory_usage(self):
        """
        Test that sharding with gradients leads to a decrease in memory usage by roughly the amount of memory occupied
        by fully constructed flat parameter and its gradient.
        """
        model = self.create_gpt_model()
        fsdp_model = self.wrap_with_custom_fsdp(model)

        root_unit = fsdp_model
        block_unit = fsdp_model.module.transformer.h[0]
        root_unit._gather(include_grads=True)
        block_unit._gather(include_grads=True)

        self.check_memory_change(root_unit, 'shard', include_grads=True)
        self.check_memory_change(block_unit, 'shard', include_grads=True)

    def test_fsdp_meta_init_memory_peak(self):
        """
        Test that the peak memory usage during FSDP construction 
        when starting from meta device never exceeds the size of the full flat parameter.
        """
        original_create_and_shard = CustomFSDP._create_and_shard_flat_param

        def patched_create_and_shard(fsdp_self, param_init_fn):
            base_alloc = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

            original_create_and_shard(fsdp_self, param_init_fn)
            
            local_shard = fsdp_self.local_shard
            shard_numel = local_shard.numel()
            grad_numel = local_shard.grad.numel()
            element_size = local_shard.element_size()
            expected_peak_alloc = (base_alloc + (shard_numel * fsdp_self.world_size + shard_numel + grad_numel) * element_size) /(2**20)
            actual_peak_alloc = (torch.cuda.max_memory_allocated())/(2**20)
            tolerance = expected_peak_alloc * 0.1
            self.assertAlmostEqual(actual_peak_alloc, expected_peak_alloc, delta=tolerance)

            expected_final_alloc = (base_alloc + (shard_numel + grad_numel) * element_size)/(2**20)
            actual_final_alloc = (torch.cuda.memory_allocated())/(2**20)
            tolerance = expected_final_alloc * 0.1
            self.assertAlmostEqual(actual_final_alloc, expected_final_alloc, delta=tolerance)
        
        CustomFSDP._create_and_shard_flat_param = patched_create_and_shard

        model = self.create_gpt_model('meta')
        fsdp_model = self.wrap_with_custom_fsdp(model)
    
    def test_numerical_correctness(self):
        """
        Test numerical correctness of CustomFSDP vs. native FSDP across forward pass,
        backward pass, gradient accumulation, and multiple optimizer steps.
        """
        config = GPTConfig(n_layer=1, vocab_size=50304)
        batch_size = 2
        micro_steps = 5
        steps = 3
        lr = 3e-4

        custom_model = self.create_gpt_model()
        custom_fsdp = self.wrap_with_custom_fsdp(custom_model)
        custom_optim = torch.optim.AdamW(custom_fsdp.parameters(), lr=lr)

        native_model = self.create_gpt_model()
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Block},
        )
        native_fsdp = FSDP(
            native_model,
            process_group=dist.group.WORLD,
            auto_wrap_policy=auto_wrap_policy,
            use_orig_params=True,
            param_init_fn=native_model._init_weights
        )
        native_optim = torch.optim.AdamW(native_fsdp.parameters(), lr=lr)


        for step in range(steps):
            custom_optim.zero_grad()
            native_optim.zero_grad()
            for micro_step in range(micro_steps):
                input_tensor = torch.randint(0, config.vocab_size, (batch_size, config.block_size), device=self.device)
                custom_logits = custom_fsdp(input_tensor)
                native_logits = native_fsdp(input_tensor)
                # 1. Test forward pass
                if step == 0 and micro_step == 0:
                    self.assertTrue(torch.equal(custom_logits, native_logits))

                # 2. Test single backward pass
                custom_loss = custom_logits.sum()
                native_loss = native_logits.sum()
                custom_loss.backward()
                native_loss.backward()
                if step == 0 and micro_step == 0:
                    self.compare_parameters(native_fsdp, custom_fsdp, compare_grads=True)
            # 3. Test gradient accumulation
            if step == 0:
                self.compare_parameters(native_fsdp, custom_fsdp, compare_grads=True)

            custom_optim.step()
            native_optim.step()

            # 4. Test final parameter data matches after multiple steps
            self.compare_parameters(native_fsdp, custom_fsdp)


    

if __name__ == "__main__":

    unittest.main()
