#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 16
#define BLOCK_SIZE 1024

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void matrix_multiply(const float *x, const float *w, float *y, int M, int C, int H, int W, int K) {

    __shared__ float tile1[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile2[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    float Y_val = 0;

    //for unrolling

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int X_col = C * K * K;
    const int Y_col = H_out * W_out;
    int unroll_w, unroll_p, unroll_c, unroll_q, unroll_h, unroll_row, unroll_col;


    for(int i = 0; i < ceil(1.0*X_col/TILE_WIDTH); i++) {
        int col_idx = i*TILE_WIDTH + threadIdx.x;

        // Load tile 1 - w
        if(col_idx < X_col){
          tile1[threadIdx.y][threadIdx.x] = w[row*X_col+col_idx];
        }
        else {
          tile1[threadIdx.y][threadIdx.x] = 0;
        }
        //ignore otherwise

        int row_idx = i*TILE_WIDTH+threadIdx.y;

        // Load tile 2 - x
        if(row_idx < X_col){
            //begin unroll
            unroll_row = (row_idx * Y_col + col) / Y_col;
            unroll_col = (row_idx * Y_col + col) % Y_col;
            unroll_q = unroll_row % K;
            unroll_p = (unroll_row / K) % K;
            unroll_c = (unroll_row / K) / K;
            unroll_w = unroll_col % W_out;
            unroll_h = unroll_col / W_out;
            //load second tile
            tile2[threadIdx.y][threadIdx.x] = x[unroll_c * H * W + (unroll_h + unroll_p) * W + (unroll_w + unroll_q)];
        } 
        else {
         tile2[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        // Add tile information to sum
        for(int j = 0; j < TILE_WIDTH; j++) {
            Y_val += tile1[threadIdx.y][j] * tile2[j][threadIdx.x];
        }
        __syncthreads();

    }// end of loop thru tiles

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

    // loop thru all batch elements
    for (int b = B; b--; ){
        
        dim3 gridDim(ceil(H*W*1.0/TILE_WIDTH), ceil(M*1.0/TILE_WIDTH));
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        float * X = x.dptr_ + b*C*H*W;
        float * W_ptr = w.dptr_;
        float * Y = y.dptr_ + b*M*H_out*W_out;
        matrix_multiply<<<gridDim, blockDim>>>(X, W_ptr, Y, M, C, H, W, K);
    }

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
