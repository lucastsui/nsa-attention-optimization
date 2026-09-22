"""Instrumentation control: a safe run or an intentionally invalid device read.

Run --bad only under Compute Sanitizer in its own process. The failed control
must report an invalid global read and a nonzero sanitizer exit status; it is
not a failure in the NSA candidate. Disabling PyTorch's memory cache makes the
CUDA allocation boundary match the tensor boundary for this test.
"""
import argparse
import os
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
import torch
import triton
import triton.language as tl


@triton.jit
def parallel_nsa_fwd_kernel_probe(x, y, BAD: tl.constexpr):
    offset = 16 if BAD else 0
    tl.store(y, tl.load(x + offset))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bad', action='store_true')
    a = p.parse_args()
    x = torch.ones(16, device='cuda')
    y = torch.empty(1, device='cuda')
    kernel = parallel_nsa_fwd_kernel_probe[(1,)](x, y, BAD=a.bad)
    print('Compiled probe kernel: '+kernel.name, flush=True)
    try:
        torch.cuda.synchronize()
        if not a.bad:
            assert y.item() == 1
    except RuntimeError:
        if not a.bad:
            raise
        print('Faulting control reported a CUDA execution error, as expected.', flush=True)
    print('Probe launched; the sanitizer log and exit code determine the outcome.', flush=True)


if __name__ == '__main__':
    main()
