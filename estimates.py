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

def get_gpt_ops(dims: ModelDimensions) -> Dict[str, Dict[str, Union[Dict[str, Union[int, List[Tuple[int, ...]], Dict[str, List[Tuple[int, ...]]]]], bool]]]:
    b, s, h, a, d, L, V = dims.b, dims.s, dims.h, dims.a, dims.d, dims.L, dims.V
    # input_dims - tensors that need to be moved from HBM to on-chip memory
    # output-dims - tensors reruned by operation
    # activation_dims - tensors produced in the forward pass that are needed in the backward pass to compute gradients. 
    return {
        'embeddings_sum': {
            'forward': {
                'flops': 2 * b * s * h,
                'input_dims': [(b, s, h), (s, h)],
                'output_dims': [(b, s, h)], #output
                'activation_dims': {
                    'float32': [], # no tensor needed for backward pass 
                    'float16': [] # n/a - both tensors are float32
                } 
            },
            'backward': {
                'flops': 0, #no flops involved or just assignment
                'input_dims': []
            },
            'is_matmul': False,
            'is_per_layer': False,
            'autocasts_to_float32': True
        },
        'pre_attn_layer_norm': {
            'forward': {
                'flops': 5 * b * s * h,
                'input_dims': [(b, s, h), (b, s), (b, s), (h,), (h,)], #input, mean, variance, gamma (param), beta (param)
                'output_dims': [(b, s, h)], #output
                'activation_dims': {
                    'float32': [(b, s, h), (b,s), (b,s)], #input, mean, variance
                    'float16': [] #n/a - only done float32
                }
            },
            'backward': {
                'flops': 14 * b * s * h,
                'input_dims': [(b, s, h), (b, s, h), (b, s), (b, s), (h,), (h,)] #output.grad, input, mean, variance, gamma (param), beta (param)
            },
            'is_matmul': False,
            'is_per_layer': True,
            'autocasts_to_float32': True
        },
        'qkv_proj': {
            'forward': {
                'flops': 6 * b * s * h ** 2,
                'input_dims': [(b, s, h), (h, 3 * h)], #input, weight
                'output_dims': [(b, s, 3*h)], #output
                'activation_dims': {
                    'float32': [(b, s, h)], #input
                    'float16': [(b, s, h), (h, 3 * h)] #input, weight
                }
            },
            'backward': {
                'flops': 12 * b * s * h ** 2,
                'input_dims': [(b, s, 3 * h), (b, s, 3 * h), (b, s, h), (h, 3 * h)], #output.grad, output.grad, input, weight
            },
            'is_matmul': True,
            'is_per_layer': True,
            'autocasts_to_float32': False
        },
        'qkT': {
            'forward': {
                'flops': 2 * b * a * s ** 2 * d,
                'input_dims': [(b, a, s, d), (b, a, d, s)], #input (copy of q slice), input (copy of k slice)
                'output_dims': [(b, a, s, s)], #output
                'activation_dims': {
                    'float32': [(b, a, s, d), (b, a, d, s)], #input (copy of q slice), input (copy of k slice)
                    'float16': [(b, a, s, d), (b, a, d, s)] #input (copy of q slice), input (copy of k slice)
                }
            },
            'backward': {
                'flops': 4 * b * a * s ** 2 * d,
                'input_dims': [(b, a, s, s), (b, a, s, s), (b, a, s, d), (b, a, d, s)] #output.grad, output.grad, input (copy of q slice), input (copy of k slice)
            },
            'is_matmul': True,
            'is_per_layer': True,
            'autocasts_to_float32': False
        },
        'scaling': {
            'forward': {
                'flops': b * a * s ** 2, 
                'input_dims': [(b, a, s, s)], #input
                'output_dims': [(b, a, s, s)], #output
                'activation_dims': {
                    'float32': [], # no tensor needed for backward pass
                    'float16': [] # no tensor needed for backward pass
                }
            },
            'backward': {
                'flops': b * a * s ** 2,
                'input_dims': [(b, a, s, s)] #output.grad
            },
            'is_matmul': False,
            'is_per_layer': True,
            'autocasts_to_float32': False
        },
        'softmax': {
            'forward': {
                'flops': 5 * b * a * s ** 2, #assumes finding max involves O(n) flops
                'input_dims': [(b, a, s, s)], #input
                'output_dims': [(b, a, s, s)], #output
                'activation_dims': {
                    'float32': [(b, a, s, s)], #output
                    'float16': [] #n/a - only done float32
                }
            },
            'backward': {
                'flops': 4 * b * a * s ** 2,
                'input_dims': [(b, a, s, s), (b, a, s, s)] #ouput.grad, output
            },
            'is_matmul': False,
            'is_per_layer': True,
            'autocasts_to_float32': True
        },
        'att_mm_v': {
            'forward': {
                'flops': 2 * b * a * s ** 2 * d,
                'input_dims': [(b, a, s, s), (b, a, s, d)], #input, #input (copy of v slice)
                'output_dims': [(b, a, s, d)], #output
                'activation_dims': {
                    'float32': [(b, a, s, d)], #input (copy of v slice) only
                    'float16': [(b, a, s, s), (b, a, s, d)] #input (need float16 copy of softmax output), #input (copy of v slice)
                }
            },
            'backward': {
                'flops': 4 * b * a * s ** 2 * d,
                'input_dims': [(b, a, s, d), (b, a, s, d), (b, a, s, s), (b, a, s, d)] #output.grad, output.grad, input, input (copy of v slice)
            },
            'is_matmul': True,
            'is_per_layer': True,
            'autocasts_to_float32': False
        },
        'output_proj': {
            'forward': {
                'flops': 2 * b * s * h ** 2,
                'input_dims': [(b, s, h), (h, h)], #input, weight
                'output_dims': [(b, s, h)], #output
                'activation_dims': {
                    'float32': [(b, s, h)], #input
                    'float16': [(b, s, h), (h, h)] #input, weight
                }
            },
            'backward': {
                'flops': 4 * b * s * h ** 2,
                'input_dims': [(b, s, h), (b, s, h), (b, s, h), (h, h)] #output.grad, output.grad, input, weight
            },
            'is_matmul': True,
            'is_per_layer': True,
            'autocasts_to_float32': False
        },
        'post_attn_residual': {
            'forward' : {
                'flops': b * s * h,
                'input_dims': [(b, s, h), (b, s, h)],
                'output_dims': [(b, s, h)], #output
                'activation_dims': {
                    'float32': [], # no tensor needed for backward pass
                    'float16': [] # n/a - at least 1 tensor is float32 irrespective of full precision or mixed precision training
                }
            },
            'backward': {
                'flops': 0,
                'input_dims': []
            },
            'is_matmul': False,
            'is_per_layer': True,
            'autocasts_to_float32': True
        },
        'pre_mlp_layer_norm': {
            'forward': {
                'flops': 5 * b * s * h,
                'input_dims': [(b, s, h), (b, s), (b, s), (h,), (h,)], #input, mean, variance, gamma (param), beta (param)
                'output_dims': [(b, s, h)], #output
                'activation_dims': {
                    'float32': [(b, s, h), (b,s), (b,s)], #input, mean, variance
                    'float16': [] #n/a - only done float32
                }
            },
            'backward': {
                'flops': 14 * b * s * h,
                'input_dims': [(b, s, h), (b, s, h), (b, s), (b, s), (h,), (h,)] #output.grad, input, mean, variance, gamma (param), beta (param)
            },
            'is_matmul': False,
            'is_per_layer': True,
            'autocasts_to_float32': True
        },
        'mlp_up_proj': {
            'forward': {
                'flops': 8 * b * s * h ** 2,
                'input_dims': [(b, s, h), (h, 4 * h)], #input, weight
                'output_dims': [(b, s, 4*h)], #output
                'activation_dims': {
                    'float32': [(b, s, h)], #input
                    'float16': [(b, s, h), (h, 4 * h)] #input, weight
                }
            },
            'backward': {
                'flops': 16 * b * s * h ** 2,
                'input_dims': [(b, s, 4 * h), (b, s, 4 * h), (b, s, h), (h, 4 * h)] #output.grad, output.grad, input, weight
            },
            'is_matmul': True,
            'is_per_layer': True,
            'autocasts_to_float32': False
        },
        'gelu': {
            'forward': {
                'flops': 32 * b * s * h,
                'input_dims': [(b, s, 4 * h)], #input
                'output_dims': [(b, s, 4*h)], #output
                'activation_dims': {
                    'float32': [(b, s, 4*h)], #input
                    'float16': [(b, s, 4*h)] #input
                }
            },
            'backward': {
                'flops': 72 * b * s * h,
                'input_dims': [(b, s, 4 * h), (b, s, 4 * h)] #output.grad, input
            },
            'is_matmul': False,
            'is_per_layer': True,
            'autocasts_to_float32': False
        },
        'mlp_down_proj': {
            'forward': {
                'flops': 8 * b * s * h ** 2,
                'input_dims': [(b, s, 4 * h), (4 * h, h)], #input, weight
                'output_dims': [(b, s, h)], #output
                'activation_dims': {
                    'float32': [(b, s, 4*h)], #input
                    'float16': [(b, s, 4*h), (4 * h, h)] #input, weight
                }
            },
            'backward': {
                'flops': 16 * b * s * h ** 2,
                'input_dims': [(b, s, h), (b, s, h), (b, s, 4 * h), (4 * h, h)] #output.grad, output.grad, input, weight
            },
            'is_matmul': True,
            'is_per_layer': True,
            'autocasts_to_float32': False
        },
        'post_mlp_residual': {
            'forward' : {
                'flops': b * s * h,
                'input_dims': [(b, s, h), (b, s, h)],
                'output_dims': [(b, s, h)], #output
                'activation_dims': {
                    'float32': [], # no tensor needed for backward pass
                    'float16': [] # n/a - at least 1 tensor is float32 irrespective of full precision or mixed precision training
                }
            },
            'backward': {
                'flops': 0,
                'input_dims': []
            },
            'is_matmul': False,
            'is_per_layer': True,
            'autocasts_to_float32': True
        },
        'final_layer_norm': {
            'forward': {
                'flops': 5 * b * s * h,
                'input_dims': [(b, s, h), (b, s), (b, s), (h,), (h,)],#input, mean, variance, gamma (param), beta (param)
                'output_dims': [(b, s, h)], #output
                'activation_dims': {
                    'float32': [(b, s, h), (b,s), (b,s)], #input, mean, variance
                    'float16': [] #n/a - only done float32
                }
            },
            'backward': {
                'flops': 14 * b * s * h,
                'input_dims': [(b, s, h), (b, s, h), (b, s), (b, s), (h,), (h,)]#output.grad, input, mean, variance, gamma (param), beta (param)
            },
            'is_matmul': False,
            'is_per_layer': False,
            'autocasts_to_float32': True
        },
        'lm_head': {
            'forward': {
                'flops': 2 * b * s * h * V,
                'input_dims': [(b, s, h), (h, V)], #input, weight
                'output_dims': [(b, s, V)], #output
                'activation_dims': {
                    'float32': [(b, s, h)], #input
                    'float16': [(b, s, h), (h,V)] #input, weight
                }
            },
            'backward': {
                'flops': 4 * b * s * h * V,
                'input_dims': [(b, s, V), (b, s, V), (b, s, h), (h, V)] #output.grad, output.grad, input, weight
            },
            'is_matmul': True,
            'is_per_layer': False,
            'autocasts_to_float32': False
        },
        'log_softmax': {
            'forward': {
                'flops': 5 * b * s * V,
                'input_dims': [(b, s, V)], #input
                'output_dims': [(b, s, V)], #output
                'activation_dims': {
                    'float32': [(b,s, V)], #output
                    'float16': [] #n/a - only done float32
                }
            },
            'backward': {
                'flops': 4 * b * s * V,
                'input_dims': [(b, s, V), (b, s, V)] #output.grad, output
            },
            'is_matmul': False,
            'is_per_layer': False,
            'autocasts_to_float32': True
        }
    }

def get_activation_memory(dims: ModelDimensions, precision: PrecisionType) -> Dict[str, int]:
    ops = get_gpt_ops(dims)
    activation_memory = {}
    for name, op_info in ops.items():
        float32_count = sum(math.prod(dim) for dim in op_info['forward']['activation_dims']['float32'])
        float16_count = sum(math.prod(dim) for dim in op_info['forward']['activation_dims']['float16'])
        if precision == PrecisionType.FULL or (precision == PrecisionType.MIXED and op_info['autocasts_to_float32']):
            activation_memory[name] = float32_count * 4
        else:
            activation_memory[name] = float16_count * 2
    return activation_memory


def calculate_peak_memory(dims: ModelDimensions, precision: PrecisionType) -> float:
    param_counts = get_param_counts(dims)    
    P = sum(param_counts.values())
    
    param_mem = 4 * P
    gradient_mem = 4 * P # assumes scheme under which grads persists (i.e grads initialised to zero or gradient accumulation
    optimizer_mem = 8 * P
    buffer_mem = 4 * (dims.s ** 2) * dims.L
    statically_allocated_mem = param_mem + gradient_mem + optimizer_mem + buffer_mem
    
    activation_mem = get_activation_memory(dims, precision)
    ops = get_gpt_ops(dims)
    dynamically_allocated_mem = 0
    for name, mem in activation_mem.items():
        dynamically_allocated_mem += mem * (dims.L if ops[name]['is_per_layer'] else 1)
    
    peak_mem_gb = (statically_allocated_mem + dynamically_allocated_mem) / (2**30)
    
    return peak_mem_gb

def get_input_memory(dims: ModelDimensions, precision: PrecisionType) -> Dict[str, Dict[str, int]]:
    ops = get_gpt_ops(dims)
    input_memory = {}

    for name, op_info in ops.items():
        fwd_count = sum(math.prod(dim) for dim in op_info['forward']['input_dims'])
        bwd_count = sum(math.prod(dim) for dim in op_info['backward']['input_dims'])

        bytes_per_element = 4 if precision == PrecisionType.FULL or (precision == PrecisionType.MIXED and op_info['autocasts_to_float32']) else 2
        
        input_memory[name] = {
            'forward': fwd_count * bytes_per_element,
            'backward': bwd_count * bytes_per_element
        }
    return input_memory

def estimate_mops_latency(dims: ModelDimensions, precision: PrecisionType, accelerator: str) -> Dict[str, Dict[str, float]]:
    input_memory = get_input_memory(dims, precision)
    
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
            "cuda": {
                'fp32': 67e12,
                'bf16': 134e12, #same as fp16
            },
            "tensor": {
                'tf32': 495e12,
                'bf16': 989e12, #same as fp16
            }
        },
        "A100": {
            "cuda": {
                'fp32': 19.5e12,
                'bf16': 39e12, #note: not the same as fp16 (=78e12)
            },
            "tensor": {
                'tf32': 156e12,
                'bf16': 312e12, #same as fp16
            }
        }
    }

    latencies = {}
    for name, op_info in ops.items():
        if op_info['is_matmul']:
            throughput = throughputs[accelerator]['tensor']['tf32' if precision == PrecisionType.FULL else 'bf16']
        elif op_info['autocasts_to_float32']:
            throughput = throughputs[accelerator]['cuda']['fp32']
        else:
            throughput = throughputs[accelerator]['cuda']['fp32' if precision == PrecisionType.FULL else 'bf16']

        fwd_latency = op_info['forward']['flops'] / (throughput * efficiency)
        bwd_latency = op_info['backward']['flops'] / (throughput * efficiency)

        latencies[name] = {
            'forward': fwd_latency,
            'backward': bwd_latency
        }
    return latencies

def estimate_combined_latency(dims: ModelDimensions, precision: PrecisionType, accelerator: str, efficiency: float = 0.8) -> Dict[str, Dict[str, float]]:
    mops_latency = estimate_mops_latency(dims, precision, accelerator)
    flops_latency = estimate_flops_latency(dims, precision, accelerator, efficiency)

    latencies = {}  
    for name in mops_latency.keys():
        fwd_latency = max(mops_latency[name]['forward'], flops_latency[name]['forward'])
        bwd_latency = max(mops_latency[name]['backward'], flops_latency[name]['backward'])
        latencies[name] = {
            'forward': fwd_latency,
            'backward': bwd_latency
        }
    return latencies

def main():
    models = [
        ("125M", 12, 12, 768),
        # ("350M", 24, 16, 1024),
        # ("1.3B", 24, 32, 2048),
        # ("2.7B", 32, 32, 2560),
        # ("6.7B", 32, 32, 4096),
        # ("13B", 40, 40, 5120), #GPT-3 paper says 5140
        # ("30B", 48, 56, 7168),
        # ("66B", 64, 72, 9216),
        # ("175B", 96, 96, 12288),
    ]

    BATCH_SIZE = 12
    SEQ_LENGTH = 1024
    VOCAB_SIZE = 50304

    precision = PrecisionType.MIXED
    accelerator = "A100"

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

        # print(model_name, calculate_peak_memory(dims, precision))

if __name__ == '__main__':
    main()