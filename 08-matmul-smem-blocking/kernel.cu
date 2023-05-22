#define BM 16
#define BN 16
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

    float sum = 0;
    for (int k = 0; k < K; k += BK) {
        _m = ty;  
        _k = tx;  
        a_block[_m][_k] = a[(m+_m)*K + k+_k];

        _k = ty;
        _n = tx;
        b_block[_k][_n] = b[(k+_k)*N + n+_n];

        __syncthreads(); 

        _m = ty;
        _n = tx;
        for (int _k = 0; _k < BK; _k++) {
            sum += a_block[_m][_k] * b_block[_k][_n];
        } 

        __syncthreads();
    }

    _m = ty;
    _n = tx;
    c[(m+_m)*N + n+_n] = sum;
}