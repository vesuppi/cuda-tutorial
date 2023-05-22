#define BM 64
#define BN 64
#define BK 16

#define TM 4
#define TN 4

extern "C" __global__
void kernel(float* a, float* b, float* c, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int m = blockIdx.y * BM;   // starting index of the block
    int n = blockIdx.x * BN;   
    int _m, _n, _k;  // inner dimensions

    __shared__ float a_block[BM][BK];
    __shared__ float b_block[BK][BN];

    float As[TM];
    float Bs[TN];
    float Cs[TM][TN] = {0};

    float sum = 0;
    for (int k = 0; k < K; k += BK) {
        _m = ty;  
        _k = tx;  
        for (_m = ty; _m < BM; _m += blockDim.y) {
            a_block[_m][_k] = a[(m+_m)*K + k+_k];
        }
        
        _k = ty;
        _n = tx;
        for (_n = tx; _n < BN; _n += blockDim.x) {
            b_block[_k][_n] = b[(k+_k)*N + n+_n];
        }
        
        __syncthreads(); 

        _m = ty;
        _n = tx;
        for (int _k = 0; _k < BK; _k++) {
            //sum += a_block[_m][_k] * b_block[_k][_n];

            int i = 0;
            for (_m = ty; _m < BM; _m += blockDim.y) {
                int j = 0;
                for (_n = tx; _n < BN; _n += blockDim.x) {
                    Cs[i][j] += a_block[_m][_k] * b_block[_k][_n];

                    j++;
                }
                i++;
            }
        } 

        __syncthreads();
    }

    int i = 0;
    for (_m = ty; _m < BM; _m += blockDim.y) {
        int j = 0;
        for (_n = tx; _n < BN; _n += blockDim.x) {
            c[(m+_m)*N + n+_n] = Cs[i][j];

            j++;
        }
        i++;
    }
}