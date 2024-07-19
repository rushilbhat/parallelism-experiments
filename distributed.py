# distributed.py
import torch
import torch.distributed as dist

class Bucket:
    def __init__(self):
        self.parameters = []
        self.gradients = []
        self.size = 0 #in bytes
        self.grad_count = 0

    def add_param(self, param):
        self.parameters.append(param)
        self.size += param.numel() * param.element_size()

    def reset(self):
        self.gradients.clear()


class CustomDDP(torch.nn.Module):
    def __init__(self, module, process_group, bucket_cap_mb=25): 
        super().__init__()
        self.module = module
        self.process_group = process_group
        self.bucket_cap_mb = bucket_cap_mb
        self.buckets = []
        self.futures = []
        self.require_backward_grad_sync = True
        self._create_buckets()
        self._register_hooks()

    def _create_buckets(self):
        params = reversed(list(self.module.parameters()))
        current_bucket = Bucket()

        for param in params:
            if param.requires_grad:
                param_size = param.numel() * param.element_size()
                if current_bucket.size + param_size > self.bucket_cap_mb * 1024 * 1024:
                    self.buckets.append(current_bucket)
                    current_bucket = Bucket()
                current_bucket.add_param(param)
        self.buckets.append(current_bucket)
        
    def _create_hook(self, bucket, param):
        def hook(grad):
            if self.require_backward_grad_sync:
                bucket.gradients.append(grad)
                if len(bucket.gradients) == len(bucket.parameters):
                    self._reduce_bucket(bucket)
        return hook

    def _register_hooks(self):
        for bucket in self.buckets:
            for param in bucket.parameters:
                hook = self._create_hook(bucket, param)
                param.register_hook(hook)

    def _reduce_bucket(self, bucket):
        flat_grads = torch.cat([grad.flatten() for grad in bucket.gradients])
        future = dist.all_reduce(flat_grads, group=self.process_group, async_op=True)
        self.futures.append((future, bucket))

    def set_require_backward_grad_sync(self, require_sync):
        self.require_backward_grad_sync = require_sync

    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _finalize_backward(self):
        # Wait for all asynchronous operations to complete
        world_size = dist.get_world_size(self.process_group)
        for future, bucket in self.futures:
            future.wait()
            flat_grads = future.result()
            flat_grads[0].div_(world_size)
            self._unflatten_and_copy(flat_grads, bucket)
        self.futures.clear()

    def _unflatten_and_copy(self, flat_grads, bucket):
        offset = 0
        for param in bucket.parameters:
            numel = param.numel()
            param.grad = flat_grads[0][offset:offset+numel].view_as(param)
            offset += numel
        bucket.reset()