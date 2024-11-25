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

def get_param_counts(ops: Dict, dims: ModelDimensions) -> Dict[str, int]:
    counts = {}
    for name, op_info in ops.items():
        counts[name] = sum(math.prod(param) for param in op_info['params'])
    return counts

def get_gpt_ops(dims: ModelDimensions) -> Dict[str, Dict[str, Union[Dict[str, Union[int, List[Tuple[int, ...]], Dict[str, List[Tuple[int, ...]]]]], bool]]]:
    b, s, h, a, d, L, V = dims.b, dims.s, dims.h, dims.a, dims.d, dims.L, dims.V
    # input_dims - tensors that need to be moved from HBM to on-chip memory
    # output-dims - tensors returned by operation
    # activation_dims - tensors produced in the forward pass that are needed in the backward pass to compute gradients. 
    return {
        'wte': {
            'params': [], #weight shared with lm_head
            'forward': {
                'flops': 0,
                'input_dims': [],
                'output_dims': [],
                'activation_dims': {
                    'float32': [],
                    'float16': [] 
                } 
            },
            'backward': {
                'flops': 0, 
                'input_dims': []
            },
            'is_matmul': False,
            'is_per_layer': False,
            'autocasts_to_float32': True
        },
        'wpe': {
            'params': [(s, h)],
            'forward': {
                'flops': 0,
                'input_dims': [],
                'output_dims': [],
                'activation_dims': {
                    'float32': [],
                    'float16': [] 
                } 
            },
            'backward': {
                'flops': 0, 
                'input_dims': []
            },
            'is_matmul': False,
            'is_per_layer': False,
            'autocasts_to_float32': True
        },
        'embeddings_sum': {
            'params': [],
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
            'params': [(h,), (h,)],
            'forward': {
                'flops': 5 * b * s * h,
                'input_dims': [(b, s, h), (h,), (h,)], #input, gamma (param), beta (param)
                'output_dims': [(b, s, h)], #output
                'activation_dims': {
                    'float32': [(b, s, h), (b,s), (b,s)], #input, mean, variance
                    'float16': [] #n/a - only done float32
                }
            },
            'backward': {
                'flops': 13 * b * s * h,
                'input_dims': [(b, s, h), (b, s, h), (b, s), (b, s), (h,), (h,)] #output.grad, input, mean, variance, gamma (param), beta (param)
            },
            'is_matmul': False,
            'is_per_layer': True,
            'autocasts_to_float32': True
        },
        'qkv_proj': {
            'params': [(h, 3 * h)],
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
            'params': [],
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
            'params': [],
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
            'params': [],
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
            'params': [],
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
            'params': [(h, h)],
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
            'params': [],
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
            'params': [(h,), (h,)],
            'forward': {
                'flops': 5 * b * s * h,
                'input_dims': [(b, s, h), (h,), (h,)], #input, gamma (param), beta (param)
                'output_dims': [(b, s, h)], #output
                'activation_dims': {
                    'float32': [(b, s, h), (b,s), (b,s)], #input, mean, variance
                    'float16': [] #n/a - only done float32
                }
            },
            'backward': {
                'flops': 13 * b * s * h,
                'input_dims': [(b, s, h), (b, s, h), (b, s), (b, s), (h,), (h,)] #output.grad, input, mean, variance, gamma (param), beta (param)
            },
            'is_matmul': False,
            'is_per_layer': True,
            'autocasts_to_float32': True
        },
        'mlp_up_proj': {
            'params': [(h, 4 * h)],
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
            'params': [],
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
            'params': [(4 * h, h)],
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
            'params': [],
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
            'params': [(h,), (h,)],
            'forward': {
                'flops': 5 * b * s * h,
                'input_dims': [(b, s, h), (h,), (h,)], #input, gamma (param), beta (param)
                'output_dims': [(b, s, h)], #output
                'activation_dims': {
                    'float32': [(b, s, h), (b,s), (b,s)], #input, mean, variance
                    'float16': [] #n/a - only done float32
                }
            },
            'backward': {
                'flops': 13 * b * s * h,
                'input_dims': [(b, s, h), (b, s, h), (b, s), (b, s), (h,), (h,)]#output.grad, input, mean, variance, gamma (param), beta (param)
            },
            'is_matmul': False,
            'is_per_layer': False,
            'autocasts_to_float32': True
        },
        'lm_head': {
            'params': [(h, V)],
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
            'params': [],
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

def get_activation_memory(ops: Dict, dims: ModelDimensions, precision: PrecisionType) -> Dict[str, int]:
    activation_memory = {}
    for name, op_info in ops.items():
        float32_count = sum(math.prod(dim) for dim in op_info['forward']['activation_dims']['float32'])
        float16_count = sum(math.prod(dim) for dim in op_info['forward']['activation_dims']['float16'])
        if precision == PrecisionType.FULL or (precision == PrecisionType.MIXED and op_info['autocasts_to_float32']):
            activation_memory[name] = float32_count * 4
        else:
            activation_memory[name] = float16_count * 2
    return activation_memory


def calculate_peak_memory(ops: Dict, dims: ModelDimensions, precision: PrecisionType, return_components: bool = False) -> float:
    param_counts = get_param_counts(ops, dims)    
    P = sum(count * (dims.L if ops[name]['is_per_layer'] else 1) for name, count in param_counts.items())
    
    param_mem = 4 * P
    gradient_mem = 4 * P # assumes scheme under which grads persists (i.e grads initialised to zero or gradient accumulation
    optimizer_mem = 8 * P
    buffer_mem = 4 * (dims.s ** 2) * dims.L
    statically_allocated_mem = param_mem + gradient_mem + optimizer_mem + buffer_mem
    
    activation_mem = get_activation_memory(ops, dims, precision)
    dynamically_allocated_mem = 0
    for name, mem in activation_mem.items():
        dynamically_allocated_mem += mem * (dims.L if ops[name]['is_per_layer'] else 1)
    
    peak_mem = statically_allocated_mem + dynamically_allocated_mem
    
    if return_components:
        return {
            'peak': peak_mem/2**30,
            'staically_allocated': statically_allocated_mem/2**30,
            'dynamically_allocated': dynamically_allocated_mem/2**30
        }
    return peak_mem

def get_input_memory(ops: Dict, dims: ModelDimensions, precision: PrecisionType) -> Dict[str, Dict[str, int]]:
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

def estimate_mops_latency(ops: Dict, dims: ModelDimensions, precision: PrecisionType, accelerator: str) -> Dict[str, Dict[str, float]]:
    input_memory = get_input_memory(ops, dims, precision)
    
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


def estimate_flops_latency(ops: Dict, dims: ModelDimensions, precision: PrecisionType, accelerator: str, efficiency: float = 0.8) -> Dict[str, Dict[str, float]]:

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

def estimate_combined_latency(ops: Dict, dims: ModelDimensions, precision: PrecisionType, accelerator: str, efficiency: float = 0.8) -> Dict[str, Dict[str, float]]:
    mops_latency = estimate_mops_latency(ops, dims, precision, accelerator)
    flops_latency = estimate_flops_latency(ops, dims, precision, accelerator, efficiency)

    latencies = {}  
    for name in mops_latency.keys():
        fwd_latency = max(mops_latency[name]['forward'], flops_latency[name]['forward'])
        bwd_latency = max(mops_latency[name]['backward'], flops_latency[name]['backward'])
        latencies[name] = {
            'forward': fwd_latency,
            'backward': bwd_latency
        }
    return latencies

def estimate_total_time(ops: Dict, dims: ModelDimensions, precision: PrecisionType, accelerator: str, efficiency: float = 0.8) -> Dict[str, float]:
    latency = estimate_combined_latency(ops, dims, precision, accelerator, efficiency)
    total_time_fwd = total_time_bwd = 0
    for name, op_latency in latency.items():
        total_time_fwd += op_latency['forward'] * (dims.L if ops[name]['is_per_layer'] else 1)
        total_time_bwd += op_latency['backward'] * (dims.L if ops[name]['is_per_layer'] else 1)

    return {'forward': total_time_fwd, 'backward': total_time_bwd}

def estimate_collective_comm_time(collective_op: str, accelerator: str, world_size: int, comm_volume: int) -> Tuple[float, float]:
    # base latency + per step latency * no. steps + amount of data / memory bandwidth * no. steps

    bandwidths = {
        "H100": {
            "inter": 450e9, #unidirectional
            "intra": 8*48e9 #unidirectional
        },
        "A100": {
            "inter": 300e9, #unidirectional
            "intra": 8*24e9 #unidirectional
        }
    }

    num_gpus_per_node = 8
    nNodes = math.ceil(world_size / num_gpus_per_node)
    nSteps = (world_size - 1) * (2 if collective_op == "all-reduce" else 1)#all-reduce = reduce-scatter + all-gather
    
    #default values from comm_analysis.py
    baseLat = 6.6e-6
    intraLat = 1e-6
    interLat = 2.7e-6

    #not sure why each step is split into interStep and intraStep when each step contains both inter- and intra-node communication
    #in comm_analysis.py and https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc

    #nInterSteps = 2 * nNodes if world_size > num_gpus_per_node else 0
    #nIntraSteps = nSteps - nInterSteps
    #latency = (baseLat if world_size > 1 else 0) + nIntraSteps * intraLat + nInterSteps * interLat
    
    latency  = (baseLat if world_size > 1 else 0) + nSteps * (interLat if nNodes > 1 else intraLat) #following https://github.com/NVIDIA/nccl-tests/issues/123
    effective_bandwidth = bandwidths[accelerator]["inter" if world_size > 1 else "intra"]
    transport_time = nSteps * comm_volume / (world_size * effective_bandwidth)
    
    return latency, transport_time
    # return latency + transport_time

def enumerate_world_size_microbatch_pairs(batch_size: int) -> List[Tuple[int, int]]:
    return [(n, math.ceil(batch_size/n)) for n in range(1, batch_size+1) if n == 1 or (math.ceil(batch_size/n) < math.ceil(batch_size/(n-1)))]

def get_full_ops_list(ops: Dict, dims: ModelDimensions) -> List[str]:
    ops = list(ops.keys())

    layer_start_idx = next(i for i, op in enumerate(ops) if op == 'pre_attn_layer_norm')
    layer_end_idx = next(i for i, op in enumerate(ops) if op == 'post_mlp_residual')

    return ops[:layer_start_idx] + ops[layer_start_idx:layer_end_idx+1] * dims.L + ops[layer_end_idx+1:] #dims.L

def create_buckets(all_params: List[Tuple[str, Tuple[int, ...]]], bucket_cap: int) -> List[List[Tuple[str, Tuple[int, ...]]]]:
    non_zero_params = [(name, param) for name, param in all_params if param != (0,)]
    buckets = [[]]
    for name, param in reversed(non_zero_params):
        if sum(math.prod(dim)*4 for _, dim in buckets[-1]) < bucket_cap:
            buckets[-1].append((name, param))
        else:
            buckets.append([(name, param)])

    return buckets #in order of backward pass

# "bucket_cap = 768: (768, 50304), (768,), (768,), (3072, 768), (768, 3072), (768,), (768,), (768, 768), (768, 2304), (768,), (768,), (1024, 768)"
# "bucket_cap = 768+1 to 768*2: (768, 50304), (768,)+(768,), (3072, 768), (768, 3072), (768,)+(768,), (768, 768), (768, 2304), (768,)+(768,), (1024, 768)"
# "bucket_cap = 768*2+1 to 768**2+768*2: (768, 50304), (768,)+(768,)+(3072, 768), (768, 3072), (768,)+(768,)+(768, 768), (768, 2304), (768,)+(768,)+(1024, 768)"
# "bucket_cap = 768**2+768*2+1 to 768*3072: (768, 50304), (768,)+(768,)+(3072, 768), (768, 3072), (768,)+(768,)+(768, 768)+(768, 2304), (768,)+(768,)+(1024, 768)"
# "bucket_cap = 768*3072+1 to 768+768*3072 == 768+768**2+768*2304: (768, 50304), (768,)+(768,)+(3072, 768), (768, 3072)+(768,), (768,)+(768, 768)+(768, 2304), (768,)+(768,)+(1024, 768)"
# "bucket_cap = 768+768*3072+1 == 768+768**2+768*2304+1 to : (768, 50304), (768,)+(768,)+(3072, 768), (768, 3072)+(768,)+(768,), (768, 768)+(768, 2304)+(768,)+(768,), (1024, 768)"

#1. Go through all params and group them by bucket_cap
#2. Find the smallest (or second smallest if smallest is last bucket), increment by 4 and set as new bucket cap
#3. Repeat until bucket cap = model size
def enumerate_bucket_caps(all_params: List[Tuple[str, Tuple[int, ...]]]) -> List[int]:
    bucket_caps = [min(math.prod(p) for _, p in all_params if p != (0,)) * 4]
    model_size = sum(math.prod(p) for _, p in all_params) * 4

    while bucket_caps[-1] < model_size:
        buckets = create_buckets(all_params, bucket_caps[-1]+4)
        bucket_sizes = [sum(math.prod(p) for _, p in bucket) * 4 for bucket in buckets]
        #case 1: more than 1 bucket, smallest bucket isn't last / case 2: more than 1 bucket, smallest bucket is last / case 3: only 1 bucket
        next_cap = min(bucket_sizes[:-1] if len(bucket_sizes) > 1 and bucket_sizes.index(min(bucket_sizes)) == len(bucket_sizes)-1 
                      else bucket_sizes)
        bucket_caps.append(next_cap)

    return bucket_caps

def estimate_total_time_with_bucketed_gradient_reduction(ops: Dict, dims: ModelDimensions, precision: PrecisionType, accelerator: str, all_params: List[Tuple[str, Tuple[int, ...]]], bucket_cap: int, world_size: int, efficiency: float = 0.8, return_components: bool= False) -> Dict[str, float]:
    latency = estimate_combined_latency(ops, dims, precision, accelerator, efficiency)
    full_ops_list = get_full_ops_list(ops, dims)
    t_fwd = sum(latency[name]['forward'] for name in full_ops_list)

    buckets = create_buckets(all_params, bucket_cap)
    initial_bucket = buckets[0]
    idx = 0
    t_bwd_initial_bucket = 0
    while idx < len(initial_bucket):
        op_name = full_ops_list.pop(-1)
        t_bwd_initial_bucket += latency[op_name]['backward']
        if initial_bucket[idx][0] == op_name:
            idx += len(ops[op_name]['params'])
    t_bwd_remaining= sum(latency[op_name]['backward'] for op_name in full_ops_list)
    t_comm_buckets = [estimate_collective_comm_time("all-reduce", accelerator, world_size, sum(math.prod(p) for _, p in bucket) * 4) for bucket in buckets]
    if len(t_comm_buckets) > 1:
        t_comm_remaining = sum(lat + transport for lat, transport in t_comm_buckets[:-1])
        t_comm_final_bucket = sum(t_comm_buckets[-1])
    else:
        t_comm_remaining = sum(t_comm_buckets[0])
        t_comm_final_bucket = 0
    
    t_total = t_fwd + t_bwd_initial_bucket + max(t_bwd_remaining, t_comm_remaining) + t_comm_final_bucket
    latency = t_comm_buckets[0][0]

    if return_components:
        return {
            'total': round(t_total, 12),
            'forward': round(t_fwd, 12),
            'backward_initial_bucket': round(t_bwd_initial_bucket, 12),
            'backward_remaining': round(t_bwd_remaining, 12),
            'comm_remaining': round(t_comm_remaining, 12),
            'comm_final_bucket': round(t_comm_final_bucket, 12),
            'num_comm_steps': len(t_comm_buckets) - 1,
            'latency': t_comm_buckets[0][0]
        }

    return round(t_total, 12)

def find_optimal_bucket_cap(ops: Dict, dims: ModelDimensions, precision: PrecisionType, accelerator: str, all_params: List[Tuple[str, Tuple[int, ...]]], world_size: int, bucket_caps: List[int]) -> Tuple[float, int]:
    min_time, bucket_cap = min(
        (estimate_total_time_with_bucketed_gradient_reduction(ops, dims, precision, accelerator, all_params, bucket_cap, world_size), bucket_cap) 
        for bucket_cap in bucket_caps
    )

    return min_time, bucket_cap

def analyse_bucket_caps(ops: Dict, dims: ModelDimensions, precision: PrecisionType, accelerator: str, all_params: List[Tuple[str, Tuple[int, ...]]], world_size: int, bucket_caps: List[int]) -> Tuple[float, int]:
    print(f"{'Bucket cap':<15} {'Time':<20} {'Fwd':<23} {'Bwd_init':<23} {'Bwd_rem':<23} {'Comm_rem':<23} {'Comm_final':<23} {'Bwd_rem < Comm_rem':<25} {'Cum_comm_lat < Bwd_rem':<30} {'No. comm steps':<20} {'Comm_latency':<20}")

    for bucket_cap in bucket_caps:
        timing = estimate_total_time_with_bucketed_gradient_reduction(ops, dims, precision, accelerator, all_params, bucket_cap, world_size, return_components=True)
        cumulative_comm_latency = timing['num_comm_steps'] * timing['latency']
        print(f"{bucket_cap:<15} {timing['total']:<20} {timing['forward']:<23} {timing['backward_initial_bucket']:<23} {timing['backward_remaining']:<23} {timing['comm_remaining']:<23} {timing['comm_final_bucket']:<23} {str(timing['backward_remaining'] < timing['comm_remaining']):<25} {str(cumulative_comm_latency < timing['backward_remaining']):<30} {timing['num_comm_steps']:<20} {timing['latency']}") 

def estimate_total_time_with_fsdp(ops: Dict, dims: ModelDimensions, precision: PrecisionType, accelerator: str, world_size: int, return_components: bool=False):
    root_unit_ops = ['wte', 'wpe', 'embeddings_sum', 'final_layer_norm', 'lm_head', 'log_softmax']
    latency = estimate_combined_latency(ops, dims, precision, accelerator)

    t_bwd_init = sum(latency[op]['backward'] for op in ['log_softmax', 'lm_head', 'final_layer_norm'])

    block_size = sum(math.prod(p) for op, op_info in ops.items() 
                    if op not in root_unit_ops 
                    for p in op_info['params']) * 4
    t_bwd_block = sum(latency[op]['backward'] for op in ops if op not in root_unit_ops)
    t_rs_block = sum(estimate_collective_comm_time("reduce-scatter", accelerator, world_size, block_size))

    t_bwd_embeddings = sum(latency[op]['backward'] for op in ['embeddings_sum', 'wpe', 'wte'])
    root_size = sum(math.prod(p) for op in root_unit_ops for p in ops[op]['params']) * 4
    t_rs_root = sum(estimate_collective_comm_time("reduce-scatter", accelerator, world_size, root_size))

    time = t_bwd_init + (dims.L - 1) * max(t_rs_block, t_bwd_block) + max(t_rs_block, t_bwd_embeddings) + t_rs_root

    if return_components:
        return {
            'total': round(time, 12),
            'bwd_init': round(t_bwd_init, 12),
            'rs_block': round(t_rs_block, 12),
            'bwd_block': round(t_bwd_block, 12),
            'bwd_embeddings': round(t_bwd_embeddings, 12),
            'rs_root': round(t_rs_root, 12)
        }

    return time

def main():
    models = [
        ("125M", 12, 12, 768, 2**19),
        ("350M", 24, 16, 1024, 2**19),
        ("1.3B", 24, 32, 2048, 2**20),
        ("2.7B", 32, 32, 2560, 2**20),
        ("6.7B", 32, 32, 4096, 2**21),
        ("13B", 40, 40, 5120, 2**21), #GPT-3 paper says 5140
        # ("30B", 48, 56, 7168),
        # ("66B", 64, 72, 9216),
        ("175B", 96, 96, 12288, 2**21 + 2**20 + 2**16),
    ]

    SEQ_LENGTH = 2048
    VOCAB_SIZE = 50304

    precision = PrecisionType.MIXED
    accelerator = "H100"

    for model_name, num_layers, num_heads, hidden_dim, batch_size_in_toks in models:
        batch_size = batch_size_in_toks // SEQ_LENGTH
        ws_mb_pairs = enumerate_world_size_microbatch_pairs(batch_size)

        for world_size, microbatch_size in ws_mb_pairs:
            dims = ModelDimensions(
                a=num_heads,
                b=microbatch_size,
                d=hidden_dim // num_heads,
                h=hidden_dim,
                L=num_layers,
                s=SEQ_LENGTH,
                V=VOCAB_SIZE
            )
            if world_size == 1:
                ops = get_gpt_ops(dims)
                full_ops_list = get_full_ops_list(ops, dims)
                all_params = [(op, param) for op in full_ops_list for param in (ops[op]['params'] if ops[op]['params'] else [(0,)])]
                bucket_caps = enumerate_bucket_caps(all_params)
                continue

            if world_size==2: print(f"Model: {model_name} | Model size: {sum(math.prod(p) for _, p in all_params) * 4}")
            ops = get_gpt_ops(dims)
            
            #timing breakdown for each bucket cap
            #===================================================================================================================================
            # analyse_bucket_caps(ops, dims, precision, accelerator, all_params, world_size, bucket_caps)
            # continue
            #===================================================================================================================================

            #list optimal bucket cap for each model size, world size combo
            #===================================================================================================================================
            # if world_size ==2: print(f"{'World size':<15} {'Microbatch size':<20} {'Min time':<20} {'Bucket cap':<15}")
            # min_time, min_bucket_cap = find_optimal_bucket_cap(ops, dims, precision, accelerator, all_params, world_size, bucket_caps)
            # print(f"{world_size:<15} {microbatch_size:<20} {min_time:<20} {min_bucket_cap:<15}")
            # if world_size == batch_size: print(" ")
            #===================================================================================================================================

            #timing breakdown for fsdp
            #===================================================================================================================================
            if world_size ==2: print(f"{'World size':<15} {'Microbatch size':<20} {'Time':<20} {'Bwd_init':<20} {'Rs_block':<20} {'Bwd_block':<20} {'Bwd_embeddings':<20} {'Rs_root':<20}")
            timing = estimate_total_time_with_fsdp(ops, dims, precision, accelerator, world_size, return_components=True)
            print(f"{world_size:<15} {microbatch_size:<20} {timing['total']:<20} {timing['bwd_init']:<20} {timing['rs_block']:<20} {timing['bwd_block']:<20} {timing['bwd_embeddings']:<20} {timing['rs_root']:<20}")
            if world_size == batch_size: print(" ")
            #===================================================================================================================================


if __name__ == '__main__':
    main()