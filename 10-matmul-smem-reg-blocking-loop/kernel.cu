#define BM 128
#define BN 64
#define BK 16

#define TM BM/16
#define TN BN/16

#define DX 16
#define DY 16

extern "C" __global__
void kernel(const float* a, const float* b, float* c, const uint M, const uint N, const uint K) {
    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;
    // int dx = blockDim.x;
    // int dy = blockDim.y;
    const uint m = blockIdx.y * BM;   // starting index of the block
    const uint n = blockIdx.x * BN;   
    uint _m, _n, _k;  // inner dimensions

    __shared__ float a_block[BM][BK];
    __shared__ float b_block[BK][BN];

    // float A[TM];
    // float B[TN];
    float C[TM][TN] = {0};

    uint i = 0, j = 0;


    for (int k = 0; k < K; k += BK) {
        _m = ty;  
        _k = tx; 
        for (i = 0; i < TM; i++) {
            a_block[_m+i*DY][_k] = a[(m+_m+i*DY)*K + k+_k];
        }
        
        _k = ty;
        _n = tx;
        for (j = 0; j < TN; j++) {
            b_block[_k][_n+j*DX] = b[(k+_k)*N + n +_n+j*DX];
        }
        
        __syncthreads(); 

        _m = ty;
        _n = tx;
        
        for (_k = 0; _k < BK; _k++) {
            // for (i = 0; i < TM; i++) {
            //     A[i] = a_block[_m+i*DY][_k];
            // }

            // for (j = 0; j < TN; j++) {
            //     B[j] = b_block[_k][_n+j*DX];
            // }

            // for (i = 0; i < TM; i++) {
            //     for (j = 0; j < TN; j++) {
            //         C[i][j] += A[i] * B[j];
            //     }
            // }

            for (i = 0; i < TM; i++) {
                for (j = 0; j < TN; j++) {
                    C[i][j] += a_block[_m+i*DY][_k] * b_block[_k][_n+j*DX];
                }
            }
        } 

        __syncthreads();
    }

    _m = ty;
    _n = tx;
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            c[(m+i*DY+_m)*N + n+_n+j*DX] = C[i][j];
        }
    }
}