#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 16

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


__global__ void gemm(int numXColumns, int numWRows, int numWColumns, float *X, float *W, float *Y){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    float Y_val = 0;

    if(row < numWRows && col < numXColumns)
    for(int i = 0; i < numWColumns; ++i){
      Y_val += X[row*numWColumns+i] * W[i*numXColumns+col];
    }
      Y[row*numXColumns+col] = Y_val;
}

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B,
                                const int M, const int C, const int H, const int W, const int K)
{

  // An example use of these macros:
  // float a = y4d(0,0,0,0)
  // y4d(0,0,0,0) = a

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    //OPTIMIZATION: shared memory convolution
    //BTW the reasoning behind this is in chapter 16 pg 15 of the book :D
    int X_TILE_WIDTH = TILE_WIDTH + (k - 1);
    __shared__ float x_shared[X_TILE_WIDTH][X_TILE_WIDTH];
    //TODO: move this to const memory to satisfy another optimization!
    __shared__ float k_shared[k][k];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    //int H_grid = ceil(1.0 * H_out / TILE_WIDTH);


    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int n = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
    int c, p, q;
    if (h < H_out && w < W_out) {

        float value = 0;
        // loop thru input feature maps
        for(c = 0; c < C; c++){

            //loop thru convolution matrix elements
            //p, q are indexes of the convolution matrix
            for(p = 0; p < K; p++){
                for(q = 0; q < K; q++){
                    value += x4d(n,c,h+p,w+q) * k4d(m,c,p,q);

                }
            }
        }
    y4d(n,m,h,w) = value;
   }


#undef y4d
#undef x4d
#undef k4d
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0]; //Number of elements in batch
    const int M = y.shape_[1]; //Number of output feature maps
    const int C = x.shape_[1]; //Number of input feature maps
    const int H = x.shape_[2]; //Number of output elements (height)
    const int W = x.shape_[3]; //Number of output elemetns (width)
    const int K = w.shape_[3]; //Size of convolution matrix (K x K)
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int W_grid = ceil(W_out * 1.0 / TILE_WIDTH);
    const int H_grid = ceil(H_out * 1.0 / TILE_WIDTH);
    const int Z = H_grid * W_grid;
    // Set the kernel dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // x -> number of samples in batch
    // y -> number of vertical tiles per output map
    // z -> where output tile is inside of the output map
    dim3 gridDim(B,M,Z);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim, 0>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
    gemm<<<dimGrid, dimBlock>>>(H_unroll, B, W, X_unrolled, W_unrolled, Y[n])
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
