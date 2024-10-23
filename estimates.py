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

def calculate_total_memory(dims: ModelDimensions, precision: PrecisionType) -> Dict[str, Tuple[float, str]]:
    param_counts = get_param_counts(dims)    
    P = sum(param_counts.values())
    
    param_mem = 4 * P if precision == PrecisionType.FULL else 2 * P
    gradient_mem = 4 * P if precision == PrecisionType.FULL else 2 * P
    optimizer_mem = 8 * P if precision == PrecisionType.FULL else 16 * P
    buffer_mem = 4 * (dims.s ** 2) * dims.L
    activation_mem = sum(get_activation_memory(dims, precision).values())
   
    total_mem_gb = (param_mem + gradient_mem + optimizer_mem + buffer_mem + activation_mem) / (2**30)
    return total_mem_gb

#-----------------------------------UPDATE--------------------------------------------
def get_flops(dims: ModelDimensions) -> Dict[str, int]:
    b, s, h, a, d, L, V = dims.b, dims.s, dims.h, dims.a, dims.d, dims.L, dims.V
    return {
        'embeddings_sum': 2 * b * s * h,
        'qkv_proj': 6 * b * s * h ** 2,
        'qkT': 2 * b * a * (s ** 2) * d,
        'scaling': b * a * s ** 2,
        'softmax': 5 * b * a * s ** 2,
        'att_mm_v': 2 * b * a * (s ** 2) * d,
        'output_proj': 2 * b * s * h ** 2,
        'mlp_up_proj': 8 * b * s * h ** 2,
        'gelu': 8 * b * s * h,
        'mlp_down_proj': 8 * b * s * h ** 2,
        'block_layer_norms': 2 * (5 * b * s * h),
        'block_residuals': 2 * (b * s * h),
        'final_layer_norm': 5 * b * s * h,
        'lm_head': 2 * b * s * h * V
    }

def estimate_mops_latencies(dims: ModelDimensions) -> Dict[str, float]:
    params = get_param_counts(dims)
    activations = get_activation_counts(dims)

    memory_bandwidth = 1555e9 #3.35e12
    bytes_per_element = 4
    
    latencies = {}

    for op in activations.keys():
        param_mem = params.get(op, 0) * bytes_per_element
        act_mem = activations.get(op, 0) * bytes_per_element
        latencies[op] = (param_mem + act_mem) / memory_bandwidth
        
    layer_ops = ['qkv_proj', 'qkT', 'scaling', 'softmax', 'att_mm_v', 'output_proj', 
                'mlp_up_proj', 'gelu', 'mlp_down_proj', 'block_layer_norms', 'block_residuals']
    for op in layer_ops:
        latencies[op] *= dims.L

    return latencies


def estimate_flops_latencies(dims: ModelDimensions, efficiency: float = 0.8) -> Dict[str, float]:
    flops = get_flops(dims)
    
    h100_fp32_peak = 67e12  # 494 TFLOPS for FP32
    h100_tf32_peak = 989e12  # 989 TFLOPS for TF32
    h100_bf16_peak = 1979e12
    a100_fp32_peak = 19.5e12
    a100_tf32_peak = 156e12
    a100_bf16_peak = 312e12

    # Adjust peak performance by efficiency factor
    # efficiency = 1
    h100_fp32_peak *= efficiency
    h100_tf32_peak *= efficiency
    h100_bf16_peak *= efficiency
    a100_fp32_peak *= efficiency
    a100_tf32_peak *= efficiency
    a100_bf16_peak *= efficiency

    
    latencies = {}
    
    matmul_ops = ['qkv_proj', 'qkT', 'att_mm_v', 'output_proj', 'mlp_up_proj', 'mlp_down_proj', 'lm_head']
    
    # Calculate latencies
    for op, op_flops in flops.items():
        if op in matmul_ops:
            latencies[op] = op_flops / a100_tf32_peak
        else:
            latencies[op] = op_flops / a100_fp32_peak
    
    # Multiply by number of layers for layer-specific operations
    layer_ops = ['qkv_proj', 'qkT', 'scaling', 'softmax', 'att_mm_v', 'output_proj', 
                 'mlp_up_proj', 'gelu', 'mlp_down_proj', 'block_layer_norms', 'block_residuals']
    for op in layer_ops:
        latencies[op] *= dims.L
    
    return latencies

def estimate_combined_latencies(dims: ModelDimensions, efficiency: float = 0.8) -> Dict[str, float]:
    mops_latencies = estimate_mops_latencies(dims)
    flops_latencies = estimate_flops_latencies(dims, efficiency)
    
    combined_latencies = {}
    
    for op in mops_latencies.keys():
        if op != 'total':
            combined_latencies[op] = max(mops_latencies[op], flops_latencies[op])
    
    combined_latencies['total'] = sum(combined_latencies.values())
    
    return combined_latencies


def estimate_flops_memory_arithmetic_intensities(dims: ModelDimensions):
    flops = get_flops(dims)
    params = get_param_counts(dims)
    activations = get_activation_counts(dims)

    layer_ops = ['qkv_proj', 'qkT', 'scaling', 'softmax', 'att_mm_v', 'output_proj', 
                 'mlp_up_proj', 'gelu', 'mlp_down_proj', 
                 'block_layer_norms', 'block_residuals']
    matmul_ops = ['qkv_proj', 'qkT', 'att_mm_v', 'output_proj', 
                  'mlp_up_proj', 'mlp_down_proj', 
                  'lm_head']

    # Calculate FLOPS
    matmul_flops = other_flops = 0
    for op, op_flops in flops.items():
        total_op_flops = op_flops * (dims.L if op in layer_ops else 1)
        if op in matmul_ops:
            matmul_flops += total_op_flops
        else:
            other_flops += total_op_flops
    total_flops = matmul_flops + other_flops
    
    # Calculate Memory
    bytes_per_element = 4  # Assuming 32-bit floating point numbers

    matmul_params = other_params = 0
    for op, op_params in params.items():
        total_op_params = op_params * (dims.L if op in layer_ops else 1)
        if op in matmul_ops:
            matmul_params += total_op_params
        elif op != 'wpe':
            other_params += total_op_params

    
    matmul_activations = other_activations = 0
    for op, op_activations in activations.items():
        total_op_activations = op_activations * (dims.L if op in layer_ops else 1)
        if op in matmul_ops:
            matmul_activations += total_op_activations
        else:
            other_activations += total_op_activations

    matmul_memory = (matmul_params + matmul_activations) * bytes_per_element
    other_memory = (other_params + other_activations) * bytes_per_element
    total_memory = matmul_memory + other_memory
    
    # Calculate Arithmetic Intensity (FLOPS per byte)
    matmul_intensity = matmul_flops / matmul_memory
    other_intensity = other_flops / other_memory
    total_intensity = total_flops / total_memory

    return {
        'matmul_flops': matmul_flops,
        'other_flops': other_flops,
        'total_flops': total_flops,
        'matmul_memory': matmul_memory,
        'other_memory': other_memory,
        'total_memory': total_memory,
        'matmul_intensity': matmul_intensity,
        'other_intensity': other_intensity,
        'total_intensity': total_intensity
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

def print_model_analysis(model_name, results):
    print(f"\nDetailed breakdown for {model_name} model:")
    print("\nFLOPS Breakdown:")
    print(f"Matmul FLOPS: {format_number(results['matmul_flops'])}")
    print(f"Other FLOPS: {format_number(results['other_flops'])}")
    print(f"Total FLOPS: {format_number(results['total_flops'])}")

    print("\nMemory Breakdown:")
    print(f"Matmul Memory: {format_number(results['matmul_memory'])} bytes")
    print(f"Other Memory: {format_number(results['other_memory'])} bytes")
    print(f"Total Memory: {format_number(results['total_memory'])} bytes")

    print("\nArithmetic Intensity:")
    print(f"Matmul AI: {results['matmul_intensity']:.2f} FLOPS/byte")
    print(f"Other AI: {results['other_intensity']:.2f} FLOPS/byte")
    print(f"Total AI: {results['total_intensity']:.2f} FLOPS/byte")
    print("-" * 60)


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
    SEQ_LENGTH = 2048
    VOCAB_SIZE = 50257

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

        print(calculate_memory_requirements(dims, PrecisionType.MIXED))
        import sys; sys.exit()

        
        results = estimate_flops_memory_arithmetic_intensities(dims)
        
        total_flops = format_number(results['total_flops'])
        matmul_flops = format_number(results['matmul_flops'])
        other_flops = format_number(results['other_flops'])
        total_memory = format_number(results['total_memory'])
        matmul_memory = format_number(results['matmul_memory'])
        other_memory = format_number(results['other_memory'])
        total_ai = format_number(results['total_intensity'])
        matmul_ai = format_number(results['matmul_intensity'])
        other_ai = format_number(results['other_intensity'])

            
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

