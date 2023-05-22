#define BM 16
#define BN 16
#define BK 16

#define TM 4
#define TN 4

extern "C" __global__
void kernel(float* a, float* b, float* c, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int m = blockIdx.y * BM;   // starting index of the block
    int n = blockIdx.x * BN;   
    int __m, __n, __k;  // inner dimensions

    __shared__ float a_block[BM][BK];
    __shared__ float b_block[BK][BN];

    float As[]

    float sum = 0;
    for (int k = 0; k < K; k += BK) {
        __m = ty;
        __k = tx;
        for (int _m = 0; _m < BM; _m += blockDim.y) {
            int row = m + _m + ty;
            int col = k + tx;
            a_block[m+_m+__m][__k] = a[(m+_m+__m)*K + k+__k];
        }

        __k = ty;
        __n = tx;
        for (int _n = 0; _n < BN; _n += blockDim.x) {
            int row = k + ty;
            int col = n + tx + _n;
            b_block[__k][n+_n+__n] = b[(k+__k)*N + n+_n+__n];
        }

        __syncthreads(); 

        _m = ty;
        _n = tx;
        for (int _k = 0; _k < BK; _k++) {
            sum += a_block[_m][_k] * b_block[_k][_n];
            float a0 = a_block[_m][_k];
            float a1 = a_block[(_m+blockDim.y)][_k];

            float b0 = b[k*N + n];
            float b1 = b[k*N + (blockDim.x+n)];

            sum[0] += a0 * b0;
            sum[1] += a0 * b1;
            sum[2] += a1 * b0;
            sum[3] += a1 * b1;
        } 

        __syncthreads();
    }

    _m = ty;
    _n = tx;
    c[(m+_m)*N + n+_n] = sum;
}