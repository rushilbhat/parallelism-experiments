from dataclasses import dataclass
from typing import Dict, Tuple
from enum import Enum


@dataclass
class ModelDimensions:
    a: int  # number of attention heads
    b: int  # batch size
    d: int  # head dimension
    h: int  # hidden dimension
    L: int  # number of layers
    s: int  # sequence length
    V: int  # vocabulary size

class PrecisionType(Enum):
    FULL = "full" 
    MIXED = "mixed"


def get_param_counts(dims: ModelDimensions) -> Dict[str, int]:
    s, h, L, V = dims.s, dims.h, dims.L, dims.V
    return {
        'wpe': s * h,
        'qkv_proj': 3 * h ** 2 * L,
        'output_proj': h ** 2 * L,
        'mlp_up_proj': 4 * h ** 2 * L,
        'mlp_down_proj': 4 * h ** 2 * L,
        'block_layer_norms': 2 * 2 * h * L,
        'final_layer_norm': 2 * h,
        'lm_head': h * V #wte weights shared with lm_head 
    }

def get_activation_counts(dims: ModelDimensions) -> Dict[str, int]:
    b, s, h, a, d, L, V = dims.b, dims.s, dims.h, dims.a, dims.d, dims.L, dims.V
    return {
        'embeddings_sum': (b * s * h) + (s * h),
        'qkv_proj': (b * s * h) * L,
        'qkT': 2 * (b * a * s * d) * L,
        'scaling': (b * a * s ** 2) * L,
        # 'masking': ((s ** 2) + (b * a * s ** 2)) * L,
        'softmax': (b * a * s ** 2) * L,
        'att_mm_v': ((b * a * s ** 2) + (b * a * s * d)) * L,
        'output_proj': (b * s * h) * L,
        'mlp_up_proj': (b * s * h) * L,
        'gelu': (b * s * 4 * h) * L,
        'mlp_down_proj': (b * s * 4 * h) * L,
        'block_layer_norms': 2 * ((b * s * h) + (2 * b * s)) * L,
        'block_residuals': 2 * 2 * (b * s * h) * L,
        'final_layer_norm': (b * s * h) + (2 * b * s),
        'lm_head': (b * s * h),
        'cross_entropy': (b * s * V)
    }

def get_param_memory(dims: ModelDimensions, precision: PrecisionType) -> Dict[str, int]:
    param_counts = get_param_counts(dims)
    bytes_per_element = 4 if PrecisionType.FULL else 2
    param_memory = {
        name: count * bytes_per_element 
        for name, count in param_counts.items()
    }
    return param_memory

def get_activation_memory(dims: ModelDimensions, precision: PrecisionType) -> Dict[str, int]:
    activation_counts = get_activation_counts(dims)
    matmul_ops = {'qkv_proj', 'qkT', 'att_mm_v', 'output_proj', 'mlp_up_proj', 'mlp_down_proj', 'lm_head'}

    activation_memory = {}
    for name, count in activation_counts.items():
        if precision == PrecisionType.MIXED and name in matmul_ops:
            bytes_per_element = 2
        else:
            bytes_per_element = 4
        activation_memory[name] = count * bytes_per_element

    return activation_memory

def calculate_total_memory(dims: ModelDimensions, precision: PrecisionType) -> float:
    param_counts = get_param_counts(dims)    
    P = sum(param_counts.values())
    
    param_mem = 4 * P if precision == PrecisionType.FULL else 2 * P
    gradient_mem = 4 * P if precision == PrecisionType.FULL else 2 * P
    optimizer_mem = 8 * P if precision == PrecisionType.FULL else 16 * P
    buffer_mem = 4 * (dims.s ** 2) * dims.L
    activation_mem = sum(get_activation_memory(dims, precision).values())
   
    total_mem_gb = (param_mem + gradient_mem + optimizer_mem + buffer_mem + activation_mem) / (2**30)
    return total_mem_gb

def estimate_mops_latencies(dims: ModelDimensions, precision: PrecisionType, accelerator: str) -> Dict[str, float]:
    param_mems = get_param_memory(dims, precision)
    activation_mems = get_activation_memory(dims, precision)

    memory_bandwidths = {
        "H100": 3.35e12,
        "A100": 1555e9
    }
    memory_bandwidth = memory_bandwidths[accelerator]

    latencies = {
        op: (param_mems.get(op, 0) + activation_mems[op]) / memory_bandwidth
        for op in activation_mems.keys()
    }

    return latencies

def get_flops(dims: ModelDimensions) -> Dict[str, int]:
    b, s, h, a, d, L, V = dims.b, dims.s, dims.h, dims.a, dims.d, dims.L, dims.V
    return {
        'embeddings_sum': 2 * b * s * h,
        'qkv_proj': (6 * b * s * h ** 2) * L,
        'qkT': (2 * b * a * (s ** 2) * d) * L,
        'scaling': (b * a * s ** 2) * L,
        'softmax': (5 * b * a * s ** 2) * L,
        'att_mm_v': (2 * b * a * (s ** 2) * d) * L,
        'output_proj': (2 * b * s * h ** 2) * L,
        'mlp_up_proj': (8 * b * s * h ** 2) * L,
        'gelu': (8 * b * s * h) * L,
        'mlp_down_proj': (8 * b * s * h ** 2) * L,
        'block_layer_norms': (2 * (5 * b * s * h)) * L,
        'block_residuals': (2 * (b * s * h)) * L,
        'final_layer_norm': 5 * b * s * h,
        'lm_head': 2 * b * s * h * V
    }

def estimate_flops_latencies(dims: ModelDimensions, precision: PrecisionType, accelerator: str, efficiency: float = 0.8) -> Dict[str, float]:
    flops = get_flops(dims)
    
    throughputs = {
        "H100": {
            "cuda": 67e12,         # FP32 CUDA core throughput in FLOPs per second
            "tensor": {
                PrecisionType.FULL: 989e12,  # TF32 throughput in FLOPs per second
                PrecisionType.MIXED: 1979e12, # BF16 throughput in FLOPs per second
            }
        },
        "A100": {
            "cuda": 19.5e12,       # FP32 CUDA core throughput in FLOPs per second
            "tensor": {
                PrecisionType.FULL: 156e12,   # TF32 throughput in FLOPs per second
                PrecisionType.MIXED: 312e12,  # BF16 throughput in FLOPs per second
            }
        }
    }

    cuda_core_throughput = throughputs[accelerator]['cuda']
    tensor_core_throughput = throughputs[accelerator]['tensor'][precision]

    matmul_ops = {'qkv_proj', 'qkT', 'att_mm_v', 'output_proj', 'mlp_up_proj', 'mlp_down_proj', 'lm_head'}

    latencies = {}
    for op, op_flops in flops.items():
        if op in matmul_ops:
            latency = op_flops / (tensor_core_throughput * efficiency)
        else:
            latency = op_flops / (cuda_core_throughput * efficiency)
        latencies[op] = latency
    
    return latencies

def estimate_combined_latencies(dims: ModelDimensions, precision: PrecisionType, accelerator: str, efficiency: float = 0.8) -> Dict[str, float]:
    mops_latencies = estimate_mops_latencies(dims, precision)
    flops_latencies = estimate_flops_latencies(dims, precision, accelerator, efficiency)
    
    combined_latencies = {
        op : max(mops_latencies[op], flops_latencies[op])
        for op in mops_latencies.keys()
    }    
    combined_latencies['total'] = sum(combined_latencies.values())
    
    return combined_latencies

def estimate_aggregate_flops_memory_arithmetic_intensity(dims: ModelDimensions, precision: PrecisionType) -> Dict[str, float]:
    flops = get_flops(dims)
    param_mems = get_param_memory(dims, precision)
    activation_mems = get_activation_memory(dims, precision)

    matmul_ops = {'qkv_proj', 'qkT', 'att_mm_v', 'output_proj', 'mlp_up_proj', 'mlp_down_proj', 'lm_head'}
    matmul_flops = other_flops = 0
    for op, op_flops in flops.items():
        if op in matmul_ops:
            matmul_flops += op_flops
        else:
            other_flops += op_flops
    total_flops = matmul_flops + other_flops

    matmul_memory = other_memory = 0
    for op in activation_mems.keys():
        op_memory = param_mems.get(op, 0) + activation_mems[op]
        if op in matmul_ops:
            matmul_memory += op_memory
        else:
            other_memory += op_memory
    total_memory = matmul_memory + other_memory

    matmul_ai = matmul_flops / matmul_memory
    other_ai = other_flops / other_memory
    total_ai = total_flops / total_memory

    return {
        'matmul_flops': matmul_flops,
        'other_flops': other_flops,
        'total_flops': total_flops,
        'matmul_memory': matmul_memory,
        'other_memory': other_memory,
        'total_memory': total_memory,
        'matmul_ai': matmul_ai,
        'other_ai': other_ai,
        'total_ai': total_ai
    }

def format_number(num):
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    else:
        return f"{num:.2f}"

#-----------------------------------UPDATE--------------------------------------------

def main():
    models = [
        ("125M", 12, 12, 768),
        ("350M", 24, 16, 1024),
        ("1.3B", 24, 32, 2048),
        ("2.7B", 32, 32, 2560),
        ("6.7B", 32, 32, 4096),
        ("13B", 40, 40, 5120), #GPT-3 paper says 5140
        ("30B", 48, 56, 7168),
        ("66B", 64, 72, 9216),
        ("175B", 96, 96, 12288),
    ]

    BATCH_SIZE = 8
    SEQ_LENGTH = 1024
    VOCAB_SIZE = 50257

    precision = PrecisionType.FULL
    accelerator = "A100"

    print(f"{'Model':<8} {'Tot FLOPS':<10} {'MM FLOPS':<10} {'Oth FLOPS':<10} "
          f"{'Tot Mem':<10} {'MM Mem':<10} {'Oth Mem':<10} "
          f"{'Tot AI':<8} {'MM AI':<8} {'Oth AI':<8}")
    print("-" * 110)


    for model_name, num_layers, num_heads, hidden_dim in models:
        dims = ModelDimensions(
            a=num_heads,
            b=BATCH_SIZE,
            d=hidden_dim // num_heads,
            h=hidden_dim,
            L=num_layers,
            s=SEQ_LENGTH,
            V=VOCAB_SIZE
        )
        
        results = estimate_aggregate_flops_memory_arithmetic_intensity(dims, precision)
        
        total_flops = format_number(results['total_flops'])
        matmul_flops = format_number(results['matmul_flops'])
        other_flops = format_number(results['other_flops'])
        total_memory = format_number(results['total_memory'])
        matmul_memory = format_number(results['matmul_memory'])
        other_memory = format_number(results['other_memory'])
        total_ai = format_number(results['total_ai'])
        matmul_ai = format_number(results['matmul_ai'])
        other_ai = format_number(results['other_ai'])

            
        print(f"{model_name:<8} {total_flops:<10} {matmul_flops:<10} {other_flops:<10} "
              f"{total_memory:<10} {matmul_memory:<10} {other_memory:<10} "
              f"{total_ai:<8} {matmul_ai:<8} {other_ai:<8}")

        
    
        # print(24 * dims.b*dims.s*(dims.h**2)*dims.L + 4*dims.b*dims.a*dims.s*dims.s*dims.d*dims.L + 2*dims.b*dims.s*dims.h*dims.V )
        # print(20*dims.b*dims.s*dims.h*dims.L + 6*dims.b*dims.a*dims.s*dims.s*dims.L + 7*dims.b*dims.s*dims.h)

        # import sys; sys.exit()

if __name__ == '__main__':
    main()

# dims = ModelDimensions(a=12, b=8, d=64, h=768, L=12, s=1024, V=50257)
# mops_latencies = estimate_mops_latencies(dims)
# flops_latencies = estimate_flops_latencies(dims)
# combined_latencies = estimate_combined_latencies(dims)

# print("\nComparison of latencies:")
# print(f"{'Operation':<20} {'MOPS (ms)':<15} {'FLOPS (ms)':<15} {'Combined (ms)':<15}")
# for op in combined_latencies.keys():
#     if op != 'total':
#         mops = mops_latencies.get(op, 0) * 1000
#         flops = flops_latencies.get(op, 0) * 1000
#         combined = combined_latencies[op] * 1000
#         print(f"{op:<20} {mops:<15.6f} {flops:<15.6f} {combined:<15.6f}")

# print(f"Total estimated latency: {combined_latencies['total']*1000:.6f} ms")

