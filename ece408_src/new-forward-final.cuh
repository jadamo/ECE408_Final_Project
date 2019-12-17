#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 16
#define BLOCK_SIZE 1024
#define KERN_SIZE 16000 //Max is 64 KB

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


// Calls unroll kernel from the host
// void unroll_x(float* x_unroll, int total, float* x, int C, int H, int W, int K, int b){
//     // H, C, W -> H, (C x W)
//     int H_out = H - K + 1;
//     int W_out = W - K + 1;
//
//     dim3 gridDim(ceil(C * H_out * W_out*1.0 / TILE_WIDTH), 1, 1);
//     dim3 blockDim(TILE_WIDTH, 1, 1);
//     unroll_x_kernel<<<gridDim, blockDim>>>(C, H, W, K, b, x, x_unroll);
// }

//All these parameters are needed for the #define macro :(

//__constant__ float w_const[KERN_SIZE];

__global__ void matrix_multiply(float *x, float *w, float *y, int M, int C, int H, int W, int K) {

    __shared__ float tile1[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile2[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    float Y_val = 0;

    //for unrolling

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int X_col = C * K * K;
    int Y_col = H_out * W_out;
    int idx, col_idx, row_idx, unroll_w, unroll_p, unroll_c, unroll_q, unroll_h, unroll_row, unroll_col;
    //take advantage of parallelism
    x += blockIdx.z*C*H*W;
    y += blockIdx.z*M*Y_col;

    // Loop thru tiles
    for(int i = 0; i < ceil(1.0*X_col/TILE_WIDTH); i++) {
        col_idx = i*TILE_WIDTH + threadIdx.x;
        row_idx = i*TILE_WIDTH+threadIdx.y;

        // Load w tile
        if(col_idx < X_col){
            tile1[threadIdx.y][threadIdx.x] = w[row*X_col+col_idx];
        } 
        else {
            tile1[threadIdx.y][threadIdx.x] = 0;
        }

        // Load x tile
        if(row_idx < X_col){

            idx = row_idx * Y_col + col;
            unroll_row = idx / Y_col;
            unroll_col = idx % Y_col;
            unroll_q = unroll_row % K;
            unroll_row/= K;
            unroll_p = unroll_row % K;
            unroll_c = unroll_row / K;
            unroll_w = unroll_col % W_out;
            unroll_h = unroll_col / W_out;
            //load second tile
            tile2[threadIdx.y][threadIdx.x] = x[unroll_c * H * W + (unroll_h + unroll_p) * W + (unroll_w + unroll_q)];
        }
        else {
            tile2[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for(int j = 0; j < TILE_WIDTH; j++) {
            Y_val += tile1[threadIdx.y][j] * tile2[j][threadIdx.x];
        }
        __syncthreads();
    }// <- end of loop thru tiles

    if(row < M && col < Y_col) {
        y[row * Y_col + col] = Y_val;
    }
}


/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y,
                         const mshadow::Tensor<gpu, 4, float> &x,
                         const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // w -> M x (C x K x K)
    // x -> C * K * K * (H_out * W_out)


    // Extract the tensor sions into B,M,C,H,W,K
    const int B = x.shape_[0]; //Number of elements in batch
    const int M = y.shape_[1]; //Number of output feature maps
    const int C = x.shape_[1]; //Number of input feature maps
    const int H = x.shape_[2]; //Number of output elements (height)
    const int W = x.shape_[3]; //Number of output elemetns (width)
    const int K = w.shape_[3]; //Size of convolution matrix (K x K)
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_unroll = C * K * K;
    //Mconst int H_unroll = H_out * W_out;

    //cudaMemcpyToSymbol(w_const, w.dptr_, M*W_unroll*sizeof(float));

    //float* x_unrolled;
    // float* w_unrolled;

    //int dimension = C*H*W;
    //int total = W_unroll*H_unroll;
    //cudaMalloc(&x_unrolled, total * sizeof(float));

    // Only have to unroll w once because it's the same for all batches
    // unroll_w(C, K, M, w.dptr_, w_unrolled);

    // loop thru all batch elements
    //for (int b = B; b--; ){
        //int grid = ceil(total*1.0/BLOCK_SIZE);
        //unroll_x_kernel<<<grid, BLOCK_SIZE>>>(C, H, W, K, x_unrolled, total, b*dimension + x.dptr_);

    dim3 gridDim(ceil(H_out*W_out*1.0/TILE_WIDTH), ceil(M*1.0/TILE_WIDTH), B);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    float * X = x.dptr_;
    float * W_ptr = w.dptr_;
    float * Y = y.dptr_;
    matrix_multiply<<<gridDim, blockDim>>>(X, W_ptr, Y, M, C, H, W, K);
    //}

    //free allocated memory
    //cudaFree(x_unrolled);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}


/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}

}
}

#endif
