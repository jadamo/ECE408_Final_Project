#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

// two potential tile_width values depending on input size
#define TILE_WIDTH_SMALL 16
#define TILE_WIDTH_BIG 24

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

// Combied unroll and matrix multiplication kernel
__global__ void matrix_multiply_small(const float *x, const float *w, float *y, 
                                int M, int C, int H, int W, int K) {

    __shared__ float tile1[TILE_WIDTH_SMALL][TILE_WIDTH_SMALL];
    __shared__ float tile2[TILE_WIDTH_SMALL][TILE_WIDTH_SMALL];

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    float Y_val = 0;

    //for unrolling

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int X_col = C * K * K;
    const int Y_col = H_out * W_out;
    int unroll_w, unroll_p, unroll_c, unroll_q, unroll_h;

    //take advantage of parallelism
    x += blockIdx.z*C*H*W;
    y += blockIdx.z*M*Y_col;

    for(int i = 0; i < ceil(1.0*X_col/TILE_WIDTH_SMALL); i++) {
        int col_idx = i*TILE_WIDTH_SMALL + threadIdx.x;

        if(col_idx < X_col){
          //load tile
            tile1[threadIdx.y][threadIdx.x] = w[row*X_col+col_idx];
        } else {
          tile1[threadIdx.y][threadIdx.x] = 0;
        }
        //ignore otherwise


        int row_idx = i*TILE_WIDTH_SMALL+threadIdx.y;

        if(row_idx < X_col){

          //begin unroll
         // unroll_row = (row_idx * Y_col + col) / Y_col;
         // unroll_col = (row_idx * Y_col + col) % Y_col;

         unroll_q = ((row_idx * Y_col + col) / Y_col) % K;
         unroll_p = (((row_idx * Y_col + col) / Y_col) / K) % K;
         unroll_c = (((row_idx * Y_col + col) / Y_col) / K) / K;
         unroll_w = ((row_idx * Y_col + col) % Y_col) % W_out;
         unroll_h = ((row_idx * Y_col + col) % Y_col) / W_out;
         //load second tile
         tile2[threadIdx.y][threadIdx.x] = x[unroll_c * H * W + (unroll_h + unroll_p) * W + (unroll_w + unroll_q)];
       } else {
         //nothing since we inited tiles to 0
         tile2[threadIdx.y][threadIdx.x] = 0;
       }
        __syncthreads();

        for(int j = 0; j < TILE_WIDTH_SMALL; j++) {
            Y_val += tile1[threadIdx.y][j] * tile2[j][threadIdx.x];
        }
        __syncthreads();
    }

    if(row < M && col < Y_col) {
        y[row * Y_col + col] = Y_val;
    }
}

// This is the exact same as the above code except the TILE_WIDTH is different
// Needs to be different kernel because shared mem initialization needs const values
__global__ void matrix_multiply_big(const float *x, const float *w, float *y, 
                                int M, int C, int H, int W, int K) {

    __shared__ float tile1[TILE_WIDTH_BIG][TILE_WIDTH_BIG];
    __shared__ float tile2[TILE_WIDTH_BIG][TILE_WIDTH_BIG];

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    float Y_val = 0;

    //for unrolling

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int X_col = C * K * K;
    const int Y_col = H_out * W_out;
    int unroll_w, unroll_p, unroll_c, unroll_q, unroll_h;
    
    //take advantage of parallelism
    x += blockIdx.z*C*H*W;
    y += blockIdx.z*M*Y_col;

    for(int i = 0; i < ceil(1.0*X_col/TILE_WIDTH_BIG); i++) {
        int col_idx = i*TILE_WIDTH_BIG + threadIdx.x;

        if(col_idx < X_col){
          //load tile
            tile1[threadIdx.y][threadIdx.x] = w[row*X_col+col_idx];
        } else {
          tile1[threadIdx.y][threadIdx.x] = 0;
        }
        //ignore otherwise

        int row_idx = i*TILE_WIDTH_BIG+threadIdx.y;

        if(row_idx < X_col){

          //begin unroll
         // unroll_row = (row_idx * Y_col + col) / Y_col;
         // unroll_col = (row_idx * Y_col + col) % Y_col;

         unroll_q = ((row_idx * Y_col + col) / Y_col) % K;
         unroll_p = (((row_idx * Y_col + col) / Y_col) / K) % K;
         unroll_c = (((row_idx * Y_col + col) / Y_col) / K) / K;
         unroll_w = ((row_idx * Y_col + col) % Y_col) % W_out;
         unroll_h = ((row_idx * Y_col + col) % Y_col) / W_out;
         //load second tile
         tile2[threadIdx.y][threadIdx.x] = x[unroll_c * H * W + (unroll_h + unroll_p) * W + (unroll_w + unroll_q)];
       } else {
         //nothing since we inited tiles to 0
         tile2[threadIdx.y][threadIdx.x] = 0;
       }
        __syncthreads();

        for(int j = 0; j < TILE_WIDTH_BIG; j++) {
            Y_val += tile1[threadIdx.y][j] * tile2[j][threadIdx.x];
        }
        __syncthreads();
    }

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

    // Different tile widths run quicker depending on the size of the 
    // input, so adjust depending on if the data is "big" or "small"
    float * X = x.dptr_;
    float * W_ptr = w.dptr_;
    float * Y = y.dptr_;
    if (M > 4 && C > 6){
        dim3 gridDim(ceil(H_out*W_out*1.0/TILE_WIDTH_BIG), ceil(M*1.0/TILE_WIDTH_BIG), B);
        dim3 blockDim(TILE_WIDTH_BIG, TILE_WIDTH_BIG, 1);
        matrix_multiply_big<<<gridDim, blockDim>>>(X, W_ptr, Y, M, C, H, W, K);
    }
    else{
        dim3 gridDim(ceil(H_out*W_out*1.0/TILE_WIDTH_SMALL), ceil(M*1.0/TILE_WIDTH_SMALL), B);
        dim3 blockDim(TILE_WIDTH_SMALL, TILE_WIDTH_SMALL, 1);
        matrix_multiply_small<<<gridDim, blockDim>>>(X, W_ptr, Y, M, C, H, W, K);
    }

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
