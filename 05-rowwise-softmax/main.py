import torch
from utils import bench
from cuda_kernel import kernel as cuda_softmax
from triton_kernel import kernel as triton_softmax

def torch_softmax(a, b):
    torch.softmax(a, dim=-1, out=b)

def python_softmax(a, b):
    for i in range(a.shape[0]):
        m = -1000000
        for j in range(a.shape[1]):
            if a[i,j] > m:
                m = a[i,j]
        for j in range(a.shape[1]):
            b[i,j] = torch.exp(a[i,j] - m)
        s = 0
        for j in range(a.shape[1]):
            s += b[i,j]
        for j in range(a.shape[1]):
            b[i,j] /= s   

for M in [1024*100]:
    N = 2048-1
    print(f'M: {M}, N: {N}')
    a = torch.randn(M, N, device='cuda', dtype=torch.float32)

    for f in [torch_softmax, triton_softmax]:
        b = torch.empty(M, N, device='cuda', dtype=torch.float32)
        f(a, b)
        # print(b)
        # print(torch.sum(a, dim=-1))
        assert(torch.allclose(b, torch.softmax(a, dim=-1), atol=1e-2))
        print('runtime:', bench(lambda: f(a, b)), 'ms')
    print()