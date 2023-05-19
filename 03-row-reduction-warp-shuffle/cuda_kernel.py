import os
import cupy as cp

src = open(os.path.dirname(__file__)+'/kernel.cu').read()
_kernel = cp.RawKernel(src, 'kernel', backend='nvcc', options=('-O3',))
def kernel(a, b):
    _a = cp.asarray(a)
    _b = cp.asarray(b)
    M, N = a.shape
    nthreads = 128
    nblocks = M
    _kernel(
        (nblocks,), 
        (nthreads,), 
        (_a, _b, M, N)
    )