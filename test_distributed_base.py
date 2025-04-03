import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import unittest
import torch
import torch.distributed as dist
from model import GPT, GPTConfig

torch.use_deterministic_algorithms(True)

class BaseDistributedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist.init_process_group(backend='nccl')
        cls.rank = int(os.environ['RANK'])
        cls.local_rank = int(os.environ['LOCAL_RANK'])
        cls.world_size = int(os.environ['WORLD_SIZE'])

        assert cls.world_size == 2, "These tests expect only 2 ranks."
        cls.device = f'cuda:{cls.local_rank}'
        torch.cuda.set_device(cls.device)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def create_gpt_model(self, device=None, n_layer=1, vocab_size=50304, seed=42):
        torch.cuda.manual_seed(seed)
        with torch.device(self.device if device is None else device):
            model = GPT(GPTConfig(n_layer=n_layer, vocab_size=vocab_size))
        return model

    def compare_parameters(self, model1, model2, compare_grads=False):
        for (name1, param1), (name2, param2) in zip(
                model1.named_parameters(), model2.named_parameters()
            ):
            self.assertEqual(name1, name2)
            
            # Compare parameter values
            if not compare_grads:
                self.assertTrue(torch.equal(param1, param2))
            # Compare gradients
            elif param1.grad is not None and param2.grad is not None:
                self.assertTrue(torch.equal(param1.grad, param2.grad))
            else:
                self.assertTrue(param1.grad is None and param2.grad is None) 