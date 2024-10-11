import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
from contextlib import contextmanager

@contextmanager
def unified_profiler(micro_step, idx, profiler_type='none', trace_url=None, nvtx_tag=None):
    if micro_step == 2 and idx == 0:
        if profiler_type == 'torch' and trace_url:
            profiler = profile(
                activities=[ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
                with_flops=True,
            )
            with profiler as p:
                yield
            print(p.key_averages().table(sort_by="cuda_time_total"))
            p.export_chrome_trace(trace_url)
            return
        elif profiler_type == 'nsys':
            torch.cuda.cudart().cudaProfilerStart()
        elif profiler_type == 'ncu' and nvtx_tag:
            torch.cuda.nvtx.range_push(nvtx_tag)
        
        yield

        if profiler_type == 'nsys':
            torch.cuda.cudart().cudaProfilerStop()
        elif profiler_type == 'ncu' and nvtx_tag:
            torch.cuda.nvtx.range_pop()

    else:
        yield