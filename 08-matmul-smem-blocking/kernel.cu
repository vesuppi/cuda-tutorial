#define BM 8
#define BN 32
#define BK 32

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
        // Load a tile of A (BMxBK) to shared memory
        // Note that thread block size is (BMxBN), the (BMxBK) data
        // need to be mapped to the (BMxBN) threads. We'll make BK==BN
        // to simplify the data loading
        assert(BK == BN);
        _m = ty;
        _k = tx;
        a_block[_m][_k] = a[(m+_m)*K + k+_k];

        // Load a tile of B (BKxBN) to shared memory
        // We'll make BM<=BK to simplify data loading
        assert(BM <= BK);
        
        for (int j = 0; j < BK; j+= BM) {
            b_block[ty+j][tx] = b[(k+ty+j)*N + n+tx];
        }
        // _k = ty;
        // _n = tx;
        // b_block[_k][_n] = b[(k+_k)*N + n+_n];

        // Sync threads to make sure all data is ready since
        // a warp will need to use data loaded by other warps
        __syncthreads(); 

        _m = ty;
        _n = tx;
        for (int _k = 0; _k < BK; _k++) {
            sum += a_block[_m][_k] * b_block[_k][_n];
        } 

        // Sync threads to make sure all warps have finished using the data
        // before reusing a_block and b_block for the next tile
        __syncthreads();
    }

    _m = ty;
    _n = tx;
    c[(m+_m)*N + n+_n] = sum;
}