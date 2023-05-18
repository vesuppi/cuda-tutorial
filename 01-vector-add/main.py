import torch
from utils import bench
from cuda_kernel import kernel as cuda_vec_add
from triton_kernel import kernel as triton_vec_add

def torch_vec_add(a, b, c):
    torch.add(a, b, out=c)

def python_vec_add(a, b, c):
    for i in range(a.shape[0]):
        c[i] = a[i] + b[i]

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