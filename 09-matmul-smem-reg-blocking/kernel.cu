#define BM 32
#define BN 32
#define BK 16

extern "C" __global__
void kernel(float* a, float* b, float* c, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int m = blockIdx.y * BM;   // starting index of the block
    int n = blockIdx.x * BN;   
    int _m, _n, _k;  // inner dimensions

    __shared__ float a_block[BM][BK];
    __shared__ float b_block[BK][BN];

    float sum[4] = {0};
    for (int k = 0; k < K; k += BK) {
        _m = ty;  
        _k = tx;  
        a_block[_m][_k] = a[(m+_m)*K + k+_k];
        a_block[_m+blockDim.y][_k] = a[(m+_m+blockDim.y)*K + k+_k];

        _k = ty;
        _n = tx;
        b_block[_k][_n] = b[(k+_k)*N + n+_n];
        b_block[_k][_n+blockDim.x] = b[(k+_k)*N + n+_n+blockDim.x];

        __syncthreads(); 

        _m = ty;
        _n = tx;
        for (int _k = 0; _k < BK; _k++) {
            float a0 = a_block[_m][_k];
            float a1 = a_block[_m+blockDim.y][_k];
    
            float b0 = b_block[_k][_n];
            float b1 = b_block[_k][_n+blockDim.x];
    
            sum[0] += a0 * b0;
            sum[1] += a0 * b1;
            sum[2] += a1 * b0;
            sum[3] += a1 * b1;
        } 

        __syncthreads();
    }

    _m = ty;
    _n = tx;
    c[(m+_m)*N + n+_n] = sum[0];
    c[(m+_m)*N + n+_n+blockDim.x] = sum[1];
    c[(m+_m+blockDim.y)*N + n+_n] = sum[2];
    c[(m+_m+blockDim.y)*N + n+_n+blockDim.x] = sum[3];
}