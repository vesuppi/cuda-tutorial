extern "C" __global__
void kernel(float* a, float* b, int M, int N) {
    int m = blockIdx.x;

    float sum = 0;
    // Local sum for each thread
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        sum += a[m*N + n];
    }

    // Aggregate the partial sum from all threads
    __shared__ float psums[128];  // Same as number of threads in a block

    // Store partial sums to shared memory
    psums[threadIdx.x] = sum;
    __syncthreads();

    // Reduce
    for (int step = blockDim.x/2; step > 0; step = step / 2) {
        if (threadIdx.x < step) {
            psums[threadIdx.x] += psums[threadIdx.x + step];
        }

        // Let all warps finish
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        //b[m] = psums[0];
        atomicAdd(b, psums[0]);
    }
}