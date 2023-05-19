import triton
import triton.language as tl

@triton.jit
def _kernel(a, b, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    m = tl.program_id(0)
    a_ptrs = a + m * N + tl.arange(0, BLOCK)
    a_block = tl.load(a_ptrs, mask=tl.arange(0, BLOCK) < N, other=0)
    b_block = tl.sum(a_block, axis=0)
    b_ptrs = b + m
    tl.store(b_ptrs, b_block)

def kernel(a, b):
    M, N = a.shape
    BLOCK = triton.next_power_of_2(N)
    nblocks = M
    _kernel[(nblocks,)](a, b, M, N, BLOCK)