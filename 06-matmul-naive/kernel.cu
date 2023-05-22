extern "C" __global__
void kernel(float* a, float* b, float* c, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;   
    int n = blockIdx.x * blockDim.x + threadIdx.x;  // x dim changes the fastest

    float sum = 0;
    for (int k = 0; k < K; k++) {
        sum += a[m*K + k] * b[k*N + n];
    }
    c[m*N + n] = sum;
}