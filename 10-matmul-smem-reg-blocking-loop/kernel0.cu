#define BM 32
#define BN 32
#define BK 16

#define TM 2
#define TN 2

extern "C" __global__
void kernel(float* a, float* b, float* c, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int m = blockIdx.y * BM;   // starting index of the block
    int n = blockIdx.x * BN;   
    int _m, _n, _k;  // inner dimensions

    __shared__ float a_block[BM][BK];
    __shared__ float b_block[BK][BN];

    float A[TM];
    float B[TN];
    float C[TM][TN] = {0};
    int i = 0, j = 0;
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
            A[0] = a_block[_m][_k];
            A[1] = a_block[_m+blockDim.y][_k];
    
            B[0] = b_block[_k][_n];
            B[1] = b_block[_k][_n+blockDim.x];
    
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    C[i][j] += A[i] * B[j];
                }
            }
            // C[0][0] += A[0] * B[0];
            // C[0][1] += A[0] * B[1];
            // C[1][0] += A[1] * B[0];
            // C[1][1] += A[1] * B[1];
        } 

        __syncthreads();
    }

    _m = ty;
    _n = tx;
    c[(m+_m)*N + n+_n] = C[0][0];
    c[(m+_m)*N + n+_n+blockDim.x] = C[0][1];
    c[(m+_m+blockDim.y)*N + n+_n] = C[1][0];
    c[(m+_m+blockDim.y)*N + n+_n+blockDim.x] = C[1][1];
}