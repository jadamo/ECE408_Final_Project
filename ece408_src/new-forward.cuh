#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 16

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void unroll_w_kernel(int C, int K, int M, const float* w, float* w_unroll){

  #define w4d(i3, i2, i1, i0) w[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
 
  int m = blockIdx.x*blockDim.x + threadIdx.x;
  int c = blockIdx.y*blockDim.y + threadIdx.y;

  if (m < M && c < C){
    for (int p = 0; p < K; p++){
      for (int q = 0; q < K; q++){
        int row = m;
        // Potential thing to switch if broken
        int col = c*(p*K)+q;
        w_unroll[row * C*K*K + col] = w4d(m, c, p, q);
      }
    }
  }
  #undef w4d
}

__global__ void unroll_x_kernel(int C, int H, int W, int K, int b, const float* x, float* x_unroll){
    
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    int c, s, col_out, row_out, row_unroll, col_unroll, w_base;
    int tx = blockIdx.x*blockDim.x + threadIdx.x;

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    if (tx < C * W_unroll){
        c = tx / W_unroll;
        s = tx % W_unroll;
        col_out = s / W_out;
        row_out = s % W_out;
        row_unroll = col_out * W_out + row_out;
        w_base = c*K*K;
        //loop thru convolution matrix
        for (int p = 0; p < K; p++){
            for (int q = 0; q < K; q++){
                col_unroll = w_base + (p * K) + q;
                // FIX THIS LINE!
                x_unroll[row_unroll*W_out + col_unroll] = 
                x4d(b, c, col_out+p, row_out+q);
            }
        }
    } // <- end of if
    #undef x4d
}

// Calls unroll kernel from the host
void unroll_x(int C, int H, int W, int K, int b, const float* x, float* x_unroll){
    // H, C, W -> H, (C x W)
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    dim3 gridDim(ceil(C * H_out * W_out*1.0 / TILE_WIDTH), 1, 1);
    dim3 blockDim(TILE_WIDTH, 1, 1);
    unroll_x_kernel<<<gridDim, blockDim>>>(C, H, W, K, b, x, x_unroll);
}

void unroll_w(int C, int K, int M, const float* w, float* w_unroll){

  // M, C, K, K -> M, (C x K x K)
  dim3 gridDim(ceil(M*1.0 / TILE_WIDTH), ceil(C*1.0 / TILE_WIDTH), 1);
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  unroll_w_kernel<<<gridDim, blockDim>>>(C, K, M, w, w_unroll);
}

//All these parameters are needed for the #define macro :(
__global__ void matrix_multiply(int numXColumns, const int H, const int B, const int W, 
                                const int C, const int K, const int M, 
                                int b, float *x, float *w, float *y){
    
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int m = blockIdx.z;

    int numWColumns = C*K*K;
    float Y_val = 0.0;

    if(row < B && col < numXColumns)
      for(int i = 0; i < C*K*K; i++){
        Y_val += x[row*numXColumns+i] * w[i*numWColumns+col];
      }

    //y4d(b,m,h,w)
    y4d(b, m, row, col) = Y_val;

    #undef y4d
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

    const int W_grid = ceil(W_out * 1.0 / TILE_WIDTH);
    const int H_grid = ceil(H_out * 1.0 / TILE_WIDTH);
    // Set the kernel dimensions

    // z -> number of output feature maps
    dim3 gridDim(W_grid, H_grid, M);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    float* x_unrolled;
    float* w_unrolled;
    cudaMalloc((float**) &x_unrolled, W_unroll * H_unroll * sizeof(float));
    cudaMalloc((float**) &w_unrolled, M * C*K*K * sizeof(float));

    // Only have to unroll w once because it's the same for all batches
    unroll_w(C, K, M, w.dptr_, w_unrolled);

    // loop thru all batch elements
    for (int b = 0; b < B; b++){
        unroll_x(C, H, W, K, b, x.dptr_, x_unrolled);
        cudaDeviceSynchronize();
        matrix_multiply<<<gridDim, blockDim>>>(H_unroll, H, B, W, C, K, M, b, x_unrolled, w_unrolled, y.dptr_);
        cudaDeviceSynchronize();
    }

    // Call the kernel
    //forward_kernel<<<gridDim, blockDim, 0>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    //free allocated memory
    cudaFree(x_unrolled);
    cudaFree(w_unrolled);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

// Old convolution kernel
// __global__ void forward_kernel(float *y, const float *x, const float *k, const int B, 
//                                 const int M, const int C, const int H, const int W, const int K)
// {

//   // An example use of these macros:
//   // float a = y4d(0,0,0,0)
//   // y4d(0,0,0,0) = a

//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.
//     We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     */

//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;


//     #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//     #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//     #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     int n = blockIdx.x;
//     int m = blockIdx.y;
//     int c, p, q;
//     if (h < H_out && w < W_out) {

//         float value = 0;
//         // loop thru input feature maps
//         for(c = 0; c < C; c++){

//             //loop thru convolution matrix elements
//             //p, q are indexes of the convolution matrix
//             for(p = 0; p < K; p++){
//                 for(q = 0; q < K; q++){
//                     value += x4d(n,c,h+p,w+q) * k4d(m,c,p,q);

//                 }
//             }
//         }
//     y4d(n,m,h,w) = value;
//    }


// #undef y4d
// #undef x4d
// #undef k4d
// }

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
