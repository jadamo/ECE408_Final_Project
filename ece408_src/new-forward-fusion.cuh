#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 16
#define BLOCK_SIZE 1024

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

// This file will store an implimentation of unroll + matrix multiply fusion optimization
// I'll do this one - Joe

//All these parameters are needed for the #define macro :(
__global__ void unroll_x_kernel(int C, int H, int W, int K, float* x_unroll, int total, float *X){


    int c, s;
    int tx = blockIdx.x*blockDim.x + threadIdx.x;

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    c = tx / W_unroll;  //row
    s = tx % W_unroll;  //col

    int i = c % K;
    c /= K; //come back to this
    int j = c % K;
    int k = c / K;
    int col_out = s % W_out;
    int col_out2 = s/W_out;

    x_unroll[tx] = X[k*H*W + (col_out2 + j) * W + (col_out + i)];

}

__global__ void matrix_multiply(float *x, float *w, float *y, int W_unroll, int M, int H_unroll){

    __shared__ float tile1[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile2[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    float Y_val = 0;

    for(int i = 0; i < ceil(1.0*W_unroll/TILE_WIDTH); i++){
        int col_idx = i*TILE_WIDTH + threadIdx.x;
        tile1[threadIdx.y][threadIdx.x] = 0;
        tile2[threadIdx.y][threadIdx.x] = 0;

        if(col_idx < W_unroll){
            tile1[threadIdx.y][threadIdx.x] = x[row*W_unroll+col_idx];
        }
        int row_idx = i*TILE_WIDTH+threadIdx.y;

        if(row_idx < W_unroll){
         tile2[threadIdx.y][threadIdx.x] = w[row_idx*H_unroll + col];
        }
        __syncthreads();
        for(int j = 0; j < TILE_WIDTH; j++){
            Y_val += tile1[threadIdx.y][j] * tile2[j][threadIdx.x];
        }
        __syncthreads();
    }

    if(row < M && col < H_unroll)
      for(int i = 0; i < W_unroll; i++){
          y[row*H_unroll+col] = Y_val;
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


    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0]; //Number of elements in batch
    const int M = y.shape_[1]; //Number of output feature maps
    const int C = x.shape_[1]; //Number of input feature maps
    const int H = x.shape_[2]; //Number of output elements (height)
    const int W = x.shape_[3]; //Number of output elemetns (width)
    const int K = w.shape_[3]; //Size of convolution matrix (K x K)
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int W_unroll = C * K * K;
    const int H_unroll = H_out * W_out;


    float* x_unrolled;
    // float* w_unrolled;

    int dimension = C*H*W;
    int total = W_unroll*H_unroll;
    cudaMalloc(&x_unrolled, total * sizeof(float));

    // Only have to unroll w once because it's the same for all batches
    // unroll_w(C, K, M, w.dptr_, w_unrolled);

    // loop thru all batch elements
    for (int b = B; b--; ){
        int grid = ceil(total*1.0/BLOCK_SIZE);
        unroll_x_kernel<<<grid, BLOCK_SIZE>>>(C, H, W, K, x_unrolled, total, b*dimension + x.dptr_);

        dim3 gridDim (ceil(H_unroll*1.0/TILE_WIDTH), ceil(M*1.0/TILE_WIDTH));
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        matrix_multiply<<<gridDim, blockDim>>>(w.dptr_, x_unrolled, y.dptr_+b*M*H_unroll, W_unroll, M, H_unroll);
    }

    //free allocated memory
    cudaFree(x_unrolled);

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
