import os,sys
sys.path.append('.')
import torch
import cupy as cp
from utils import bench
from cuda_kernel import kernel as cuda_matmul

def torch_matmul(a, b, c):
    torch.mm(a, b, out=c)

def python_matmul(a, b):
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            c[i,j] = 0
            for k in range(a.shape[1]):
                c[i,j] += a[i,k] * b[k,j]

for M, N, K in [(4096, 4096, 4096), (4096, 4096, 64)]:
    print(f'M: {M}, N: {N}, K: {K}')
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    print('runtimes of [torch_matmul, cuda_matmul]:')
    for f in [torch_matmul, cuda_matmul]:
        c = torch.randn(M, N, device='cuda', dtype=torch.float32)
        cp.cuda.profiler.start()
        f(a, b, c)
        cp.cuda.profiler.stop()
        #print(c)
        #print(a@b)
        assert(torch.allclose(c, a @ b, atol=1e-2))
        print(bench(lambda: f(a, b, c)), 'ms')

    break
    a_cpu, b_cpu, c_cpu = a.cpu(), b.cpu(), c.cpu()
    print('runtime of CPU version:', bench(lambda: torch_matmul(a_cpu, b_cpu, c_cpu)), 'ms')
