import torch
from utils import bench
from cuda_kernel import kernel as cuda_sum_row
from triton_kernel import kernel as triton_sum_row

def torch_sum_row(a, b):
    torch.sum(a, dim=-1, out=b)

def python_sum_row(a, b):
    for i in range(a.shape[0]):
        b[i] = 0
        for j in range(a.shape[1]):
            b[i] += a[i,j]

for M in [1024*100, 1024*1024]:
    N = 2048-1
    print(f'M: {M}, N: {N}')
    a = torch.randn(M, N, device='cuda', dtype=torch.float32)

    for f in [torch_sum_row, cuda_sum_row, triton_sum_row]:
        b = torch.empty(M, device='cuda', dtype=torch.float32)
        f(a, b)
        # print(b)
        # print(torch.sum(a, dim=-1))
        assert(torch.allclose(b, torch.sum(a, dim=-1), atol=1e-2))
        print('runtime:', bench(lambda: f(a, b)), 'ms')
    print()