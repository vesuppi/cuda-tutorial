extern "C" __global__
void kernel(float* a, float* b, int M, int N) {
    int m = blockIdx.x;

    __shared__ float smem[128];  // used for reduction
    __shared__ float a_cache

    float max_ = 0;
    // Local max for each thread
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        max_ = max(max_, a[m*N + n]);
    }

    // Store partial max to shared memory
    smem[threadIdx.x] = max_;
    __syncthreads();

    // Reduce
    for (int step = blockDim.x/2; step > 0; step = step / 2) {
        if (threadIdx.x < step) {
            smem[threadIdx.x] = max(smem[threadIdx.x], smem[threadIdx.x + step]);
        }

        // Let all warps finish
        __syncthreads();
    }

    max_ = smem[0];

    float sum = 0;
    // Local sum for each thread
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        sum += a[m*N + n];
    }

    // Store partial sums to shared memory
    smem[threadIdx.x] = sum;
    __syncthreads();

    // Reduce
    for (int step = blockDim.x/2; step > 0; step = step / 2) {
        if (threadIdx.x < step) {
            smem[threadIdx.x] += smem[threadIdx.x + step];
        }

        // Let all warps finish
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        b[m] = smem[0];
    }
}