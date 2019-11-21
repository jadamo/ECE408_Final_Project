
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH 16
#include <mxnet/base.h>
#include <stdio.h>

namespace mxnet
{
namespace op
{



__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

  // An example use of these macros:
  // float a = y4d(0,0,0,0)
  // y4d(0,0,0,0) = a

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;


  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

int X_tile_width = TILE_WIDTH + K-1;
int W_grid = ceil(W_out * 1.0/ TILE_WIDTH);
int H_grid = ceil(H_out * 1.0/ TILE_WIDTH);
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working




 int n, m, h0, w0, h_base, w_base, h, w;
 //int X_tile_width = TILE_WIDTH + K-1;
 extern __shared__ float shmem[];
 float* X_shared = &shmem[0];
 float* W_shared = &shmem[X_tile_width * X_tile_width];
 n = blockIdx.x;
 m = blockIdx.y;
 h0 = threadIdx.y;
 w0 = threadIdx.x;
 h_base = (blockIdx.z / W_grid) * TILE_WIDTH ; // vertical base out data index for the block
 w_base = (blockIdx.z % W_grid) * TILE_WIDTH ; // horizontal base out data index for the block
 h = h_base + h0;
 w = w_base + w0;
 float acc = 0.;
 int c, p, q;
 for (c = 0; c < C; c++) { // sum over all input channels
 // load weights for W [m, c,..],
// h0 and w0 used as shorthand for threadIdx.x
// and threadIdx.y
    for(int i = 0; i < ceil(K * 1.0 / TILE_WIDTH); i++) {
      int mask_w = w0 + i * TILE_WIDTH;
      int mask_h = h0 + i * TILE_WIDTH;
      if ((mask_w < K) && ( mask_h < K)) {
        int mask_w = w0 + i * TILE_WIDTH;
        int mask_h = h0 + i * TILE_WIDTH;
        W_shared[mask_h * K + mask_w] = k4d(m, c, mask_h, mask_w);

      }

    }

  __syncthreads();


 // load tile from X[n, c,…] into shared memory
 for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
  for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH) {
        if (i < H && j < W) {
          X_shared[(i - h_base) * X_tile_width + (j - w_base)] = x4d(n, c, i , j );
        }
      }
  }


 __syncthreads();
 for (p = 0; p < K; p++) {
 for (q = 0; q < K; q++) {
   if (h + p < H && w + q < W) {
     //acc += x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
     acc = acc + X_shared[(h0 + p) * X_tile_width + w0 + q ] * W_shared[p * K + q];
   }
    // acc = acc + X_shared[(h0 + p) * X_tile_width + w0 + q ] * W_shared[p * K + q];


  }
 }
 __syncthreads();
 }
 if (n <= B && m < M && h < H_out && w < W_out) {
   y4d(n, m, h, w) = acc;
 }
 __syncthreads();

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
    size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K-1) * (TILE_WIDTH + K-1) + K*K);
    // Call the kernel
    forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

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