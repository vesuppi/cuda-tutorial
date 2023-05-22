#define BM 128
#define BN 64
#define BK 16

#define TM BM/16
#define TN BN/16

extern "C" __global__
void kernel(float* a, float* b, float* c, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dx = blockDim.x;
    int dy = blockDim.y;
    int m = blockIdx.y * BM;   // starting index of the block
    int n = blockIdx.x * BN;   
    int _m, _n, _k;  // inner dimensions

    __shared__ float a_block[BM][BK];
    __shared__ float b_block[BK][BN];

    float A[TM];
    float B[TN];
    float C[TM][TN] = {0};

    int i = 0, j = 0;

    float sum = 0;
    for (int k = 0; k < K; k += BK) {
        _m = ty;  
        _k = tx; 
        for (int i = 0; i < TM; i++) {
            a_block[_m+i*dy][_k] = a[(m+_m+i*dy)*K + k+_k];
        }
        
        _k = ty;
        _n = tx;
        for (int j = 0; j < TN; j++) {
            b_block[_k][_n+j*dx] = b[(k+_k)*N + n +_n+j*dx];
        }
        
        __syncthreads(); 

        _m = ty;
        _n = tx;
        
        for (int _k = 0; _k < BK; _k++) {
            for (int i = 0; i < TM; i++) {
                A[i] = a_block[_m+dy*i][_k];
            }

            for (int j = 0; j < TN; j++) {
                B[j] = b_block[_k][_n+j*dx];
            }

            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    C[i][j] += A[i] * B[j];
                }
            }
        } 

        __syncthreads();
    }

    _m = ty;
    _n = tx;
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            c[(m+i*dy+_m)*N + n+_n+j*dx] = C[i][j];
        }
    }
}