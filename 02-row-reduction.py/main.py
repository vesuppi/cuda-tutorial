import os
import torch
import cupy as cp
from utils import bench
from triton_kernel import kernel as triton_vec_add

def torch_sum_row(a, b):
    torch.sum(a, dim=-1, out=b)

def python_sum_row(a, b):
    for i in range(a.shape[0]):
        b[i] = 0
        for j in range(a.shape[1]):
            b[i] += a[i,j]

src = open(os.path.dirname(__file__)+'/kernel.cu').read()
kernel = cp.RawKernel(src, 'kernel', backend='nvcc', options=('-O3',))
def cuda_vec_add(a, b, c):
    _a = cp.asarray(a)
    _b = cp.asarray(b)
    _c = cp.asarray(c)
    N = a.shape[0]
    nthreads = 128 
    nblocks = (N + nthreads - 1) // nthreads  # ceiling divide
    kernel(
        (nblocks,), 
        (nthreads,), 
        (_a, _b, _c, N)
    )

for shape in [1024*100, 1024*1024, 1024*1024*10, 1024*1024*10-1]:
    N = shape
    print(f'N: {N}')
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)

    for f in [torch_vec_add, cuda_vec_add, triton_vec_add]:
        c = torch.empty_like(a)
        f(a, b, c)
        assert(torch.allclose(c, a+b))
        print('runtime:', bench(lambda: f(a, b, c)), 'ms')
    print()