import estimates
from estimates import ModelDimensions, PrecisionType

def calculate_actual_flops_memory_arithmetic_intensities(dims: ModelDimensions, precision: PrecisionType):
    b, s, h, a, d, L, V = dims.b, dims.s, dims.h, dims.a, dims.d, dims.L, dims.V

    actual_matmul_flops = (24*b*s*(h**2) + 4*b*a*(s**2)*d) * L + 2*b*s*h*V
    actual_other_flops = (6*b*a*(s**2) + 20*b*s*h) * L + 6*b*s*V + 7*b*s*h + b*s
    actual_total_flops = actual_matmul_flops + actual_other_flops
    actual_matmul_memory = ((b*a*(s**2) + 12*(h**2) + 7*b*s*h + 3*b*a*s*d)*L + h*V + b*s*h) * (4 if precision == PrecisionType.FULL else 2)
    actual_other_memory = ((2*b*a*(s**2) + 10*b*s*h + 4*b*s + 4*h)*L + b*s*V + 2*b*s*h + 2*b*s + s*h + 2*h) * 4
    actual_total_memory = actual_matmul_memory + actual_other_memory
    actual_matmul_ai = actual_matmul_flops / actual_matmul_memory
    actual_other_ai = actual_other_flops / actual_other_memory
    actual_total_ai = actual_total_flops / actual_total_memory

    return {
        'matmul_flops': actual_matmul_flops,
        'other_flops': actual_other_flops,
        'total_flops': actual_total_flops,
        'matmul_memory': actual_matmul_memory,
        'other_memory': actual_other_memory,
        'total_memory': actual_total_memory,
        'matmul_ai': actual_matmul_ai,
        'other_ai': actual_other_ai,
        'total_ai': actual_total_ai
    }

#UPDATE TO INCLUDE CROSS ENTROPY!!!!!!
def calculate_and_print_percentages(results, dims: ModelDimensions):
    total_flops = results['total_flops']
    total_memory = results['total_memory']
    
    # FLOPS percentages
    matmul_flops_percent = results['matmul_flops'] / total_flops * 100
    other_flops_percent = results['other_flops'] / total_flops * 100
    
    # Memory percentages
    matmul_memory_percent = results['matmul_memory'] / total_memory * 100
    other_memory_percent = results['other_memory'] / total_memory * 100
    
    # FLOPS terms
    matmul_term1 = 24 * dims.b * dims.s * (dims.h**2) * dims.L
    matmul_term2 = 4 * dims.b * dims.a * (dims.s**2) * dims.d * dims.L
    matmul_term3 = 2 * dims.b * dims.s * dims.h * dims.V
    
    other_term1 = 6 * dims.b * dims.a * (dims.s**2) * dims.L
    other_term2 = 20 * dims.b * dims.s * dims.h * dims.L
    other_term3 = 7 * dims.b * dims.s * dims.h
    
    # Memory terms
    bytes_per_element = 4
    mem_term1 = 3 * dims.b * dims.a * (dims.s**2) * dims.L * bytes_per_element
    mem_term2 = 20 * dims.b * dims.s * dims.h * dims.L * bytes_per_element
    mem_term3 = 12 * (dims.h**2) * dims.L * bytes_per_element
    mem_term4 = 4 * dims.b * dims.s * dims.L * bytes_per_element
    mem_term5 = 4 * dims.h * dims.L * bytes_per_element
    mem_term6 = dims.h * dims.V * bytes_per_element
    mem_term7 = 3 * dims.b * dims.s * dims.h * bytes_per_element
    mem_term8 = 2 * dims.b * dims.s * bytes_per_element
    mem_term9 = dims.s * dims.h * bytes_per_element
    mem_term10 = 2 * dims.h * bytes_per_element

    
    print("\nPercentages with respect to total_flops:")
    print(f"{'Operation':<20} {'Percentage':>10}")
    print("-" * 32)
    print(f"{'matmul_flops':<20} {matmul_flops_percent:>10.2f}%")
    print(f"{'other_flops':<20} {other_flops_percent:>10.2f}%")
    
    print("\nPercentages with respect to total_memory:")
    print(f"{'Operation':<20} {'Percentage':>10}")
    print("-" * 32)
    print(f"{'matmul_memory':<20} {matmul_memory_percent:>10.2f}%")
    print(f"{'other_memory':<20} {other_memory_percent:>10.2f}%")
    
    print("\nBreakdown of flops:")
    print(f"{'Term':<30} {'Percentage':>10}")
    print("-" * 42)
    print(f"{'24*b*s*(h^2) * L':<30} {matmul_term1 / total_flops * 100:>10.2f}%")
    print(f"{'4*b*a*(s^2)*d * L':<30} {matmul_term2 / total_flops * 100:>10.2f}%")
    print(f"{'2*b*s*h*V':<30} {matmul_term3 / total_flops * 100:>10.2f}%")
    print(f"{'6*b*a*(s^2) * L':<30} {other_term1 / total_flops * 100:>10.2f}%")
    print(f"{'20*b*s*h * L':<30} {other_term2 / total_flops * 100:>10.2f}%")
    print(f"{'7*b*s*h':<30} {other_term3 / total_flops * 100:>10.2f}%")
    
    print("\nBreakdown of memory:")
    print(f"{'Term':<30} {'Percentage':>10}")
    print("-" * 42)
    print(f"{'3*b*a*(s^2)*L':<30} {mem_term1 / total_memory * 100:>10.2f}%")
    print(f"{'20*b*s*h*L':<30} {mem_term2 / total_memory * 100:>10.2f}%")
    print(f"{'12*(h^2)*L':<30} {mem_term3 / total_memory * 100:>10.2f}%")
    print(f"{'4*b*s*L':<30} {mem_term4 / total_memory * 100:>10.2f}%")
    print(f"{'4*h*L':<30} {mem_term5 / total_memory * 100:>10.2f}%")
    print(f"{'h*V':<30} {mem_term6 / total_memory * 100:>10.2f}%")
    print(f"{'3*b*s*h':<30} {mem_term7 / total_memory * 100:>10.2f}%")
    print(f"{'2*b*s':<30} {mem_term8 / total_memory * 100:>10.2f}%")
    print(f"{'s*h':<30} {mem_term9 / total_memory * 100:>10.2f}%")
    print(f"{'2*h':<30} {mem_term10 / total_memory * 100:>10.2f}%")





def compare_results(estimated, actual, metric_name):
    diff = abs(estimated - actual)
    percent_diff = (diff / actual) * 100
    print(f"{metric_name}:")
    print(f"  Estimated: {estimated:.2e}")
    print(f"  Actual: {actual:.2e}")
    print(f"  Difference: {diff:.2e} ({percent_diff:.2f}%)")
    print()

def run_tests():
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

    for model_name, num_layers, num_heads, hidden_dim in models:
        print(f"Testing {model_name} model:")
        print("-" * 40)
        
        dims = ModelDimensions(
            a=num_heads,
            b=BATCH_SIZE,
            d=hidden_dim // num_heads,
            h=hidden_dim,
            L=num_layers,
            s=SEQ_LENGTH,
            V=VOCAB_SIZE
        )

        precision = PrecisionType.MIXED
        
        estimated_results = estimates.estimate_aggregate_flops_memory_arithmetic_intensity(dims, precision)
        actual_results = calculate_actual_flops_memory_arithmetic_intensities(dims, precision)

        for metric in estimated_results.keys():
            compare_results(estimated_results[metric], actual_results[metric], metric)        
    
        print("\n")

        # calculate_and_print_percentages(actual_results, dims)


    

if __name__ == '__main__':
    run_tests()