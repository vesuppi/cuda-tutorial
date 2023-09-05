import os
import cupy as cp

src = open(os.path.dirname(__file__)+'/kernel.cu').read()
_kernel = cp.RawKernel(src, 'kernel', backend='nvcc', options=('-O3',))
def kernel(a, b, c):
    _a = cp.asarray(a)
    _b = cp.asarray(b)
    _c = cp.asarray(c)
    N = a.shape[0]
    nthreads = 128 
    nblocks = (N + nthreads - 1) // nthreads  # ceiling divide
    _kernel(
        (nblocks,), 
        (nthreads,), 
        (_a, _b, _c, N)
    )
