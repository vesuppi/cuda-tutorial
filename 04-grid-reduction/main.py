import torch
from utils import bench
from cuda_kernel import kernel as cuda_sum_all
from triton_kernel import kernel as triton_sum_all

def torch_sum_all(a, b):
    torch.sum(a, dim=None, out=b)

def python_sum_all(a, b):
    b[0] = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b[0] += a[i,j]

for M in [1024*100]:
    N = 2048
    print(f'M: {M}, N: {N}')
    a = torch.randn(M, N, device='cuda', dtype=torch.float32)

    for f in [torch_sum_all, cuda_sum_all, triton_sum_all]:
        b = torch.empty(1, device='cuda', dtype=torch.float32)
        f(a, b)
        # print(b)
        # print(torch.sum(a, dim=None))
        assert(torch.allclose(b, torch.sum(a, dim=None), atol=1e-2))
        print('runtime:', bench(lambda: f(a, b)), 'ms')
    print()