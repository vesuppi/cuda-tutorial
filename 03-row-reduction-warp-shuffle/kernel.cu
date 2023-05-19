#define FULL_MASK 0xffffffff
extern "C" __global__
void kernel(float* a, float* b, int M, int N) {
    int m = blockIdx.x;

    float sum = 0;
    // Local sum for each thread
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        sum += a[m*N + n];
    }

    // Aggregate the partial sum within a warp
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    for (int step = 32/2; step > 0; step = step / 2) {
        //if (laneId < step) {
            sum += __shfl_down_sync(FULL_MASK, sum, step);
        //}
    }

    // Aggregate across warps
    __shared__ float psums[32]; // Max number of warps per block
    if (laneId == 0) {
        psums[warpId] = sum;
    }
    __syncthreads();  // Wait for all warps to finish

    if (warpId == 0) {
        // `psum` may only be partially used. 
        // For example, if blockDim.x == 256, then only the first 8 elements 
        // actually contain the partial sum in `psum`
        if (laneId < blockDim.x / 32) {
            sum = psums[laneId];
        }
        else {
            sum = 0;
        }
        
        for (int step = 32/2; step > 0; step = step / 2) {
            //if (laneId < step) {
                sum += __shfl_down_sync(FULL_MASK, sum, step);
            //}
        }

        if (laneId == 0) {
            b[m] = sum;
        }
    }
    
}