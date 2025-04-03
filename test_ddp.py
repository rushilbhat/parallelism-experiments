import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import unittest
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from unittest.mock import patch
from distributed import CustomDDP, Reducer
from model import GPT, GPTConfig
from contextlib import nullcontext
from test_distributed_base import BaseDistributedTest

torch.use_deterministic_algorithms(True)

class TestCustomDDP(BaseDistributedTest):
    def wrap_with_custom_ddp(self, model, bucket_cap_mb=25):
        return CustomDDP(
            module=model,
            process_group=dist.group.WORLD,
            bucket_cap_mb=bucket_cap_mb
        )

    def test_hooks_registered(self):
        """
        Verify that a post-accumulate gradient hook is registered for each parameter requiring gradients.
        """
        model = self.create_gpt_model()
        model.transformer.wpe.weight.requires_grad = False
        ddp_model = self.wrap_with_custom_ddp(model)
        
        for name, p in ddp_model.module.named_parameters():
            if p.requires_grad:
                self.assertEqual(len(p._post_accumulate_grad_hooks), 1)
            else:
                self.assertIsNone(p._post_accumulate_grad_hooks)
        
    def test_bucket_creation(self):
        model = self.create_gpt_model()
        model.transformer.wpe.weight.requires_grad = False
        ddp_model = self.wrap_with_custom_ddp(model)

        # Verify only parameters with gradients are in buckets
        require_grads = list(p for p in reversed(list(model.parameters())) if p.requires_grad)
        bucketed_params = list(p for bucket in ddp_model.reducer.buckets for p in bucket.parameters.values())
        self.assertEqual(require_grads, bucketed_params)

        # Verify bucket size respects bucket_cap_mb
        reducer = ddp_model.reducer
        cap_bytes = reducer.bucket_cap_mb * 1024 * 1024
        for i, bucket in enumerate(reducer.buckets):
            total_size = bucket.size
            param_list = list(bucket.parameters.items())
            if i < len(reducer.buckets) - 1: # check n-1 buckets that they don't keep adding params after exceeding cap
                self.assertGreater(total_size, cap_bytes)
                size_without_last = sum(p.numel() * p.element_size() for _, p in param_list[:-1])
                self.assertLessEqual(size_without_last, cap_bytes)

    def test_gradient_averaging(self):
        model = self.create_gpt_model()

        batch_size = 2
        input_tensor = torch.randint(0, model.config.vocab_size, 
                                    (batch_size, model.config.block_size), 
                                    device=self.device)

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        solo_reduced_grads = {}
        for n, p in model.named_parameters():
            if p.grad is not None:
                grad = p.grad.clone()
                dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=dist.group.WORLD)
                solo_reduced_grads[n] = grad

        model.zero_grad()
        ddp_model = self.wrap_with_custom_ddp(model)
        
        output = ddp_model(input_tensor)
        loss = output.sum()
        loss.backward()

        for n, p in ddp_model.module.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.equal(p.grad, solo_reduced_grads[n]))

    def test_no_sync_skips_allreduce(self):
        model = self.create_gpt_model()
        ddp_model = self.wrap_with_custom_ddp(model)
            
        batch_size = 2
        input_tensor = torch.randint(0, model.config.vocab_size, 
                                    (batch_size, model.config.block_size), 
                                    device=self.device)
        
        with patch.object(dist, 'all_reduce', wraps=dist.all_reduce) as patched_all_reduce:
            ddp_model.set_require_backward_grad_sync(False)
            output = ddp_model(input_tensor)
            loss = output.sum()
            loss.backward()
            patched_all_reduce.assert_not_called()
        
            ddp_model.set_require_backward_grad_sync(True)
            output = ddp_model(input_tensor)
            loss = output.sum()
            loss.backward()
            patched_all_reduce.assert_called()
            self.assertEqual(patched_all_reduce.call_count, len(ddp_model.reducer.buckets))


    def test_numerical_correctness(self):
        config = GPTConfig(n_layer=1, vocab_size=50304)
        batch_size = 2
        micro_steps = 5
        steps = 3
        lr = 3e-4

        custom_model = self.create_gpt_model()
        custom_ddp = self.wrap_with_custom_ddp(custom_model)
        custom_optim = torch.optim.AdamW(custom_model.parameters(), lr=3e-4)

        native_model = self.create_gpt_model()
        native_ddp = DDP(native_model, device_ids=[self.rank])
        native_optim = torch.optim.AdamW(native_model.parameters(), lr=3e-4)
        
        for step in range(steps):
            custom_optim.zero_grad()
            native_optim.zero_grad()
            for micro_step in range(micro_steps):
                input_tensor = torch.randint(0, config.vocab_size, (batch_size, config.block_size), device=self.device)
                
                custom_ddp.set_require_backward_grad_sync(micro_step == micro_steps - 1)
                custom_logits = custom_ddp(input_tensor)

                with native_ddp.no_sync() if micro_step < micro_steps - 1 else nullcontext():
                    native_logits = native_ddp(input_tensor)

                # 1. Test forward pass
                if step == 0 and micro_step == 0:
                    self.assertTrue(torch.equal(custom_logits, native_logits))

                custom_loss = custom_logits.sum()
                native_loss = native_logits.sum()
                custom_loss.backward()
                native_loss.backward()

                # 2. Test single backward pass
                if step == 0 and micro_step == 0:
                    self.compare_parameters(native_ddp, custom_ddp, compare_grads=True)
            #3. Test gradient accumulation
            if step == 0:
                self.compare_parameters(native_ddp, custom_ddp, compare_grads=True)
            custom_optim.step()
            native_optim.step()

            # 4. Test final parameter data matches after multiple steps
            self.compare_parameters(native_ddp, custom_ddp)
                
            
if __name__ == '__main__':
    unittest.main()