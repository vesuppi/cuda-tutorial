import triton
import triton.language as tl

@triton.jit
def _kernel(a, b, c, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    a_ptrs = a + i * BLOCK + tl.arange(0, BLOCK)
    a_block = tl.load(a_ptrs)
    b_ptrs = b + i * BLOCK + tl.arange(0, BLOCK)
    b_block = tl.load(b_ptrs)
    c_block = a_block + b_block
    c_ptrs = c + i * BLOCK + tl.arange(0, BLOCK)
    tl.store(c_ptrs, c_block)

def kernel(a, b, c):
    N = a.shape[0]
    BLOCK = 128
    nblocks = (N + BLOCK - 1) // BLOCK
    _kernel[(nblocks,)](a, b, c, N, BLOCK)