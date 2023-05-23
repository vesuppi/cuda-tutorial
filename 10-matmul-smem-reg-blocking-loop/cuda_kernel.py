import os
import cupy as cp

src = open(os.path.dirname(__file__)+'/kernel.cu').read()
_kernel = cp.RawKernel(src, 'kernel', backend='nvcc', options=('-O3',))
#print(_kernel.preferred_shared_memory_carveout)
def kernel(a, b, c):
    _a = cp.asarray(a)
    _b = cp.asarray(b)
    _c = cp.asarray(c)
    M, K = a.shape
    K, N = b.shape
    nthreads = (16, 16)
    BM, BN = (128, 64)
    nblocks = (N//BN, M//BM)

    _kernel(
        nblocks, 
        nthreads, 
        (_a, _b, _c, M, N, K)
    )