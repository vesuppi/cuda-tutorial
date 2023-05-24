#define BM 128
#define BN 64
#define BK 16

#define TM BM/16
#define TN BN/16

#define DX 16
#define DY 16

extern "C" __global__
void kernel(float* a, float* b, float* c, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    //int dx = blockDim.x;
    //int dy = blockDim.y;
    int m = blockIdx.y * BM;   // starting index of the block
    int n = blockIdx.x * BN;   
    int _m, _n, _k;  // inner dimensions

    __shared__ float a_block0[BM][BK];
    __shared__ float b_block0[BK][BN];

    __shared__ float a_block1[BM][BK];
    __shared__ float b_block1[BK][BN];

    float A[TM];
    float B[TN];
    float C[TM][TN] = {0};

    int i = 0, j = 0;

    float sum = 0;
    int k = 0;


    // Load first chunk of data (k=0)
    _m = ty;  
    _k = tx; 
    for (i = 0; i < TM; i++) {
        a_block0[_m+i*DY][_k] = a[(m+_m+i*DY)*K + k+_k];
    }
    
    _k = ty;
    _n = tx;
    for (j = 0; j < TN; j++) {
        b_block0[_k][_n+j*DX] = b[(k+_k)*N + n +_n+j*DX];
    }
    
    __syncthreads(); 
    k += BK;
    
    for (; k < K; k += BK) {
        // Load next chunk of data
        _m = ty;  
        _k = tx; 
        for (i = 0; i < TM; i++) {
            a_block1[_m+i*DY][_k] = a[(m+_m+i*DY)*K + k+_k];
        }
        
        _k = ty;
        _n = tx;
        for (j = 0; j < TN; j++) {
            b_block1[_k][_n+j*DX] = b[(k+_k)*N + n +_n+j*DX];
        }
        
        // __syncthreads();    // Note how this is not needed

        // Compute on previous chunk of data
        _m = ty;
        _n = tx;
        for (int _k = 0; _k < BK; _k++) {
            for (int i = 0; i < TM; i++) {
                A[i] = a_block0[_m+DY*i][_k];
            }

            for (int j = 0; j < TN; j++) {
                B[j] = b_block0[_k][_n+j*DX];
            }

            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    C[i][j] += A[i] * B[j];
                }
            }
        } 

        __syncthreads(); 
        k += BK;
        
        if (k < K) {
            // Load next chunk of data
            _m = ty;  
            _k = tx; 
            for (i = 0; i < TM; i++) {
                a_block0[_m+i*DY][_k] = a[(m+_m+i*DY)*K + k+_k];
            }
            
            _k = ty;
            _n = tx;
            for (j = 0; j < TN; j++) {
                b_block0[_k][_n+j*DX] = b[(k+_k)*N + n +_n+j*DX];
            }

        }
                
        // __syncthreads();    // Note how this is not needed

        // Compute on previous chunk of data
        _m = ty;
        _n = tx;
        for (int _k = 0; _k < BK; _k++) {
            for (int i = 0; i < TM; i++) {
                A[i] = a_block1[_m+DY*i][_k];
            }

            for (int j = 0; j < TN; j++) {
                B[j] = b_block1[_k][_n+j*DX];
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
            c[(m+i*DY+_m)*N + n+_n+j*DX] = C[i][j];
        }
    }
}