import torch
import torch.nn as nn
import gc

def _measure_gpu_memory(stage):
    torch.cuda.synchronize()
    memory_allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
    memory_reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
    print(f"{stage}:")
    print(f"  Allocated: {memory_allocated:.2f} MB")
    print(f"  Reserved:  {memory_reserved:.2f} MB")
    
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


inpt = torch.arange(1000, dtype=torch.float32, device='cuda')
_measure_gpu_memory("After creating input tensor")
weight = torch.rand(1000, 1000, dtype=torch.float32, device='cuda')
_measure_gpu_memory("After creating weight")    

out = inpt * weight
_measure_gpu_memory("After standard multiplicaiton")    
del out
torch.cuda.empty_cache()
gc.collect()
_measure_gpu_memory("After deleting out")

out = inpt @ weight
_measure_gpu_memory("After matmul") 
del out
torch.cuda.empty_cache()
gc.collect()
_measure_gpu_memory("After deleting out")
