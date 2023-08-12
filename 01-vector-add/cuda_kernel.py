import os
import pycuda.autoprimaryctx
import numpy as np
from pycuda.compiler import SourceModule

src = open(os.path.dirname(__file__)+'/kernel.cu').read()

mod = SourceModule(src, options=['-O3'])
_kernel = mod.get_function("kernel")

def kernel(a, b, c):
    N = a.shape[0]
    nthreads = 128 * 2
    nblocks = (N + nthreads - 1) // nthreads  # ceiling divide

    _kernel(
        a, b, c, np.int32(N),
        block=(nthreads, 1, 1), 
        grid=(nblocks, 1, 1), 
    )


# import cupy as cp


# _kernel = cp.RawKernel(src, 'kernel', backend='nvcc', options=('-O3',))
# def kernel(a, b, c):
#     _a = cp.asarray(a)
#     _b = cp.asarray(b)
#     _c = cp.asarray(c)
#     N = a.shape[0]
#     nthreads = 128 
#     nblocks = (N + nthreads - 1) // nthreads  # ceiling divide
#     _kernel(
#         (nblocks,), 
#         (nthreads,), 
#         (_a, _b, _c, N)
#     )
