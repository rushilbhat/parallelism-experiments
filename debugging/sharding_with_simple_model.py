import torch
import torch.nn as nn
import sys
import gc
class Mod(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear1 = nn.Linear(size, size, bias=False)
        self.linear2 = nn.Linear(size, size, bias=False)
        # self.layernorm1 = nn.LayerNorm(size)
        # self.layernorm2 = nn.LayerNorm(size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
    def _init_weights(self, module):
        pass

# -----------------------------------------------------------------------------
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from fsdp import CustomFSDP
import os

is_distributed = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if is_distributed:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    # init_process_group(backend='gloo', rank=0, world_size=1)
    distributed_rank = int(os.environ['RANK'])
    distributed_local_rank = int(os.environ['LOCAL_RANK'])
    distributed_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{distributed_local_rank}'
    torch.cuda.set_device(device)
    master_process = distributed_rank== 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    distributed_rank = 0
    distributed_local_rank = 0
    distributed_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

input_size = 1000

with torch.device('meta'):
    model = Mod(input_size)
model = CustomFSDP(model, param_init_fn=model._init_weights, world_size=distributed_world_size, rank=distributed_rank)

if master_process: print("FORWARD")
input_tensor = torch.arange(input_size, dtype=torch.float32, device=device)
output = model(input_tensor)

