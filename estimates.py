from dataclasses import dataclass
from typing import Dict, Union, List, Tuple
from enum import Enum
import math

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

def get_gpt_ops(dims: ModelDimensions) -> Dict[str, Dict[str, Union[Dict[str, Union[int, List[Tuple[int, ...]]]], bool]]]:
    b, s, h, a, d, L, V = dims.b, dims.s, dims.h, dims.a, dims.d, dims.L, dims.V
    return {
        'embeddings_sum': {
            'forward': {
                'flops': 2 * b * s * h,
                'input_dims': [(b, s, h), (s, h)]
            },
            'backward': {
                'flops': 0, #no flops involved or just assignment
                'input_dims': []
            },
            'is_matmul': False,
            'is_per_layer': False
        },
        'pre_attn_layer_norm': {
            'forward': {
                'flops': 5 * b * s * h,
                'input_dims': [(b, s, h), (b, s), (b, s), (h,), (h,)]
            },
            'backward': {
                'flops': 14 * b * s * h,
                'input_dims': [(b, s, h), (b, s, h), (b, s), (b, s), (h,), (h,)]
            },
            'is_matmul': False,
            'is_per_layer': True        
        },
        'qkv_proj': {
            'forward': {
                'flops': 6 * b * s * h ** 2,
                'input_dims': [(b, s, h), (h, 3 * h)],
            },
            'backward': {
                'flops': 12 * b * s * h ** 2,
                'input_dims': [(b, s, 3 * h), (b, s, 3 * h), (b, s, h), (h, h)],
            },
            'is_matmul': True,
            'is_per_layer': True
        },
        'qkT': {
            'forward': {
                'flops': 2 * b * a * s ** 2 * d,
                'input_dims': [(b, a, s, d), (b, a, d, s)]
            },
            'backward': {
                'flops': 4 * b * a * s ** 2 * d,
                'input_dims': [(b, a, s, s), (b, a, s, s), (b, a, s, d), (b, a, d, s)]
            },
            'is_matmul': True,
            'is_per_layer': True
        },
        'scaling': {
            'forward': {
                'flops': b * a * s ** 2,
                'input_dims': [(b, a, s, s)]
            },
            'backward': {
                'flops': b * a * s ** 2,
                'input_dims': [(b, a, s, s)]
            },
            'is_matmul': False,
            'is_per_layer': True
        },
        'softmax': {
            'forward': {
                'flops': 5 * b * a * s ** 2, #assumes finding max involves O(n) flops
                'input_dims': [(b, a, s, s)],
            },
            'backward': {
                'flops': 4 * b * a * s ** 2,
                'input_dims': [(b, a, s, s), (b, a, s, s)]
            },
            'is_matmul': False,
            'is_per_layer': True
        },
        'att_mm_v': {
            'forward': {
                'flops': 2 * b * a * s ** 2 * d,
                'input_dims': [(b, a, s, s), (b, a, s, d)],
            },
            'backward': {
                'flops': 4 * b * a * s ** 2 * d,
                'input_dims': [(b, a, s, d), (b, a, s, d), (b, a, d, s), (b, a, s, s)] 
            },
            'is_matmul': True,
            'is_per_layer': True
        },
        'output_proj': {
            'forward': {
                'flops': 2 * b * s * h ** 2,
                'input_dims': [(b, s, h), (h, h)],

            },
            'backward': {
                'flops': 4 * b * s * h ** 2,
                'input_dims': [(b, s, h), (b, s, h), (b, s, h), (h, h)] 
            },
            'is_matmul': True,
            'is_per_layer': True
        },
        'post_attn_residual': {
            'forward' : {
                'flops': b * s * h,
                'input_dims': [(b, s, h), (b, s, h)]
            },
            'backward': {
                'flops': 0,
                'input_dims': []
            },
            'is_matmul': False,
            'is_per_layer': True
        },
        'pre_mlp_layer_norm': {
            'forward': {
                'flops': 5 * b * s * h,
                'input_dims': [(b, s, h), (b, s), (b, s), (h,), (h,)],
            },
            'backward': {
                'flops': 14 * b * s * h,
                'input_dims': [(b, s, h), (b, s, h), (b, s), (b, s), (h,), (h,)]
            },
            'is_matmul': False,
            'is_per_layer': True
        },
        'mlp_up_proj': {
            'forward': {
                'flops': 8 * b * s * h ** 2,
                'input_dims': [(b, s, h), (h, 4 * h)]
            },
            'backward': {
                'flops': 16 * b * s * h ** 2,
                'input_dims': [(b, s, 4 * h), (b, s, 4 * h), (b, s, h), (h, h)]
            },
            'is_matmul': True,
            'is_per_layer': True
        },
        'gelu': {
            'forward': {
                'flops': 32 * b * s * h,
                'input_dims': [(b, s, 4 * h)],
            },
            'backward': {
                'flops': 72 * b * s * h,
                'input_dims': [(b, s, 4 * h), (b, s, 4 * h)]
            },
            'is_matmul': False,
            'is_per_layer': True
        },
        'mlp_down_proj': {
            'forward': {
                'flops': 8 * b * s * h ** 2,
                'input_dims': [(b, s, 4 * h), (4 * h, h)]
            },
            'backward': {
                'flops': 16 * b * s * h ** 2,
                'input_dims': [(b, s, h), (b, s, h), (4 * h, h), (b, s, 4 * h)]
            },
            'is_matmul': True,
            'is_per_layer': True
        },
        'post_mlp_residual': {
            'forward' : {
                'flops': b * s * h,
                'input_dims': [(b, s, h), (b, s, h)]
            },
            'backward': {
                'flops': 0,
                'input_dims': []
            },
            'is_matmul': False,
            'is_per_layer': True
        },
        'final_layer_norm': {
            'forward': {
                'flops': 5 * b * s * h,
                'input_dims': [(b, s, h), (b, s), (b, s), (h,), (h,)],
            },
            'backward': {
                'flops': 14 * b * s * h,
                'input_dims': [(b, s, h), (b, s, h), (b, s), (b, s), (h,), (h,)]
            },
            'is_matmul': False,
            'is_per_layer': False
        },
        'lm_head': {
            'forward': {
                'flops': 2 * b * s * h * V,
                'input_dims': [(b, s, h), (h, V)],
            },
            'backward': {
                'flops': 4 * b * s * h * V,
                'input_dims': [(b, s, V), (b, s, V), (b, s, h), (h, V)]
            },
            'is_matmul': True,
            'is_per_layer': False
        },
        'cross_entropy': { #just considering log_softmax, not log_softmax + nllloss
            'forward': {
                'flops': 5 * b * s * V,
                'input_dims': [(b, s, V)]
            },
            'backward': {
                'flops': 4 * b * s * V,
                'input_dims': [(b, s, V), (b, s, V)]
            },
            'is_matmul': False,
            'is_per_layer': False
        }
    }

# def get_param_memory(dims: ModelDimensions, precision: PrecisionType) -> Dict[str, int]:
#     param_counts = get_param_counts(dims)
#     bytes_per_element = 4
#     param_memory = {
#         name: count * bytes_per_element 
#         for name, count in param_counts.items()
#     }
#     return param_memory

def get_forward_backward_input_memory(dims: ModelDimensions, precision: PrecisionType) -> Dict[str, Dict[str, int]]:
    ops = get_gpt_ops(dims)

    input_memory = {}
    for name, op_info in ops.items():
        fwd_count = sum(math.prod(dim) for dim in op_info['forward']['input_dims'])
        bwd_count = sum(math.prod(dim) for dim in op_info['backward']['input_dims'])

        bytes_per_element = 2 if precision == PrecisionType.MIXED and op_info['is_matmul'] else 4

        input_memory[name] = {
            'forward': fwd_count * bytes_per_element,
            'backward': bwd_count * bytes_per_element
        }
    return input_memory
    
def get_dynamically_allocated_memory_fwd_pass(dims: ModelDimensions, precision: PrecisionType) -> Dict[str, int]:
    ops = get_gpt_ops(dims)
    input_memory = get_forward_backward_input_memory(dims, precision)
    single_layer_dims = ModelDimensions(dims.a, dims.b, dims.d, dims.h, 1, dims.s, dims.V)
    param_counts = get_param_counts(single_layer_dims)

    allocated_memory = {}
    for name, mems in input_memory.items():
        allocated_memory[name] = mems['forward']
        # param copy is not taken for full precision training or operations that autocast to single-precision during mixed preicsion training
        if precision == PrecisionType.FULL or (precision == PrecisionType.MIXED and not ops[name]['is_matmul']): 
            allocated_memory[name] -= param_counts.get(name, 0) * 4

    return allocated_memory

def calculate_peak_memory(dims: ModelDimensions, precision: PrecisionType) -> float:
    param_counts = get_param_counts(dims)    
    P = sum(param_counts.values())
    
    param_mem = 4 * P
    gradient_mem = 4 * P # assumes scheme under which grads persists like gradient accumulation
    optimizer_mem = 8 * P
    buffer_mem = 4 * (dims.s ** 2) * dims.L

    statically_allocated_mem = param_mem + gradient_mem + optimizer_mem + buffer_mem
    dynamically_allocated_mem = sum(get_dynamically_allocated_memory_fwd_pass(dims, precision).values())

    peak_mem_gb = (statically_allocated_mem + dynamically_allocated_mem) / (2**30)

    return peak_mem_gb

def estimate_mops_latency(dims: ModelDimensions, precision: PrecisionType, accelerator: str) -> Dict[str, Dict[str, float]]:
    input_memory = get_forward_backward_input_memory(dims, precision)
    
    memory_bandwidths = {
        "H100": 3.35e12,
        "A100": 1555e9
    }
    memory_bandwidth = memory_bandwidths[accelerator]

    latencies = {
        name: {
            'forward': mems['forward'] / memory_bandwidth,
            'backward': mems['backward'] / memory_bandwidth
        }
        for name, mems in input_memory.items()
    }

    return latencies

def estimate_flops_latency(dims: ModelDimensions, precision: PrecisionType, accelerator: str, efficiency: float = 0.8) -> Dict[str, Dict[str, float]]:
    ops = get_gpt_ops(dims)

    throughputs = {
        "H100": {
            "cuda": 67e12,
            "tensor": {
                PrecisionType.FULL: 989e12,
                PrecisionType.MIXED: 1979e12,
            }
        },
        "A100": {
            "cuda": 19.5e12,
            "tensor": {
                PrecisionType.FULL: 156e12,
                PrecisionType.MIXED: 312e12,
            }
        }
    }

    cuda_core_throughput = throughputs[accelerator]['cuda']
    tensor_core_throughput = throughputs[accelerator]['tensor'][precision]

    latencies = {}
    for name, op_info in ops.items():
        if op_info['is_matmul']:
            fwd_latency = op_info['forward']['flops'] / (tensor_core_throughput * efficiency)
            bwd_latency = op_info['backward']['flops'] / (tensor_core_throughput * efficiency)
        else:
            fwd_latency = op_info['forward']['flops'] / (cuda_core_throughput * efficiency)
            bwd_latency = op_info['backward']['flops'] / (cuda_core_throughput * efficiency)

        latencies[name] = {
            'forward': fwd_latency,
            'backward': bwd_latency
        }
    return latencies

def estimate_combined_latency(dims: ModelDimensions, precision: PrecisionType, accelerator: str, efficiency: float = 0.8) -> Dict[str, Dict[str, float]]:
    mops_latency = estimate_mops_latency(dims, precision, accelerator)
    flops_latency = estimate_flops_latency(dims, precision, accelerator, efficiency)

    latencies = {}
    for op in mops_latency.keys():
        fwd_latency = max(mops_latency[op]['forward'], flops_latency[op]['forward'])
        bwd_latency = max(mops_latency[op]['backward'], flops_latency[op]['backward'])

        latencies[op] = {
            'forward': fwd_latency,
            'backward': bwd_latency
        }

    return latencies

def estimate_total_time(dims: ModelDimensions, precision: PrecisionType, acclerator: str, efficiency: float = 0.8) -> Dict[str, float]:
    latency = estimate_combined_latency(dims, precision, acclerator, efficiency)
    ops = get_gpt_ops(dims)

    layer_ops = {name for name, op_info in ops.items() if op_info['is_per_layer']}
    total_time_fwd = total_time_bwd = 0
    for name, op_latency in latency.items():
        if name in layer_ops:
            total_time_fwd += op_latency['forward'] * dims.L 
            total_time_bwd += op_latency['backward'] * dims.L
        else:
            total_time_fwd += op_latency['forward']
            total_time_bwd += op_latency['backward']

    return {'forward': total_time_fwd, 'backward': total_time_bwd}   

def estimate_aggregate_flops(dims: ModelDimensions, precision: PrecisionType) -> Dict[str, float]:
    ops = get_gpt_ops(dims)

    matmul_flops_fwd = other_flops_fwd = matmul_flops_bwd = other_flops_bwd = 0
    for name, op_info in ops.items():
        multiplier = dims.L if op_info['is_per_layer'] else 1
        if op_info['is_matmul']:
            matmul_flops_fwd += op_info['forward']['flops'] * multiplier
            matmul_flops_bwd += op_info['backward']['flops'] * multiplier

        else:
            other_flops_fwd += op_info['forward']['flops'] * multiplier
            other_flops_bwd += op_info['backward']['flops'] * multiplier

    return {
        'matmul_flops_forward': matmul_flops_fwd,
        'other_flops_forward': other_flops_fwd,
        'matmul_flops_backward': matmul_flops_bwd,
        'other_flops_backward': other_flops_bwd
    }

# def format_number(num):
#     if num >= 1e12:
#         return f"{num/1e12:.2f}T"
#     elif num >= 1e9:
#         return f"{num/1e9:.2f}B"
#     elif num >= 1e6:
#         return f"{num/1e6:.2f}M"
#     else:
#         return f"{num:.2f}"

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

    BATCH_SIZE = 1
    SEQ_LENGTH = 1024
    VOCAB_SIZE = 50304

    precision = PrecisionType.FULL
    accelerator = "A100"

    # print(f"{'Model':<8} {'Tot FLOPS':<10} {'MM FLOPS':<10} {'Oth FLOPS':<10} "
    #       f"{'Tot Mem':<10} {'MM Mem':<10} {'Oth Mem':<10} "
    #       f"{'Tot AI':<8} {'MM AI':<8} {'Oth AI':<8}")
    # print("-" * 110)


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

        total_time = estimate_total_time(dims, precision, accelerator)
        print(total_time['forward'] * 1000, total_time['backward'] * 1000)

if __name__ == '__main__':
    main()