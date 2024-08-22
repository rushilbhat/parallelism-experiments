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
    
def main():
    _measure_gpu_memory("Before creating layers on cuda device")    
    linear1 = nn.Linear(1000,1000,bias=False, device='cuda')
    _measure_gpu_memory("After creating layers on cuda device")    
    
    inpt = torch.arange(1000, dtype=torch.float32, device='cuda')
    _measure_gpu_memory("After creating input tensor")

    x = linear1(inpt)
    _measure_gpu_memory("After forwarding through linear1")
    linear1.weight.data = torch.empty(0, device='cuda')
    torch.cuda.empty_cache()
    gc.collect()
    _measure_gpu_memory("After reassigment and clearing")

    linear1.weight.data = torch.rand(1000, 1000, dtype=inpt.dtype, device='cuda')
    _measure_gpu_memory("After creating new weight tensor for linear1")
    y = linear1(x)
    _measure_gpu_memory("After forwarding through linear1 again")
    linear1.weight.data = torch.empty(0, device='cuda')
    torch.cuda.empty_cache()
    gc.collect()
    _measure_gpu_memory("After reassigment and clearing")

        
main()
torch.cuda.synchronize()
torch.cuda.empty_cache()
gc.collect()
_measure_gpu_memory("End")
