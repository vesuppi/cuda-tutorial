extern "C" __global__
void kernel(float* a, float* b, float* c, int M, int N, int K, int BM, int BN) {
    int m = blockIdx.y * BM + threadIdx.y;   
    int n = blockIdx.x * BN + threadIdx.x;  // x dim changes the fastest

    // 4 mul-add every 4 memory loads
    // before tiling it was 1 mul-add every 2 memory loads
    // should be ~2x speedup
    float sum[4] = {0};
    for (int k = 0; k < K; k++) {
        float a0 = a[m*K + k];
        float a1 = a[(m+blockDim.y)*K + k];

        float b0 = b[k*N + n];
        float b1 = b[k*N + (blockDim.x+n)];

        sum[0] += a0 * b0;
        sum[1] += a0 * b1;
        sum[2] += a1 * b0;
        sum[3] += a1 * b1;
    }
    c[m*N + n] = sum[0];
    c[m*N + (blockDim.x+n)] = sum[1];
    c[(m+blockDim.y)*N + n] = sum[2];
    c[(m+blockDim.y)*N + (blockDim.x+n)] = sum[3];
}

