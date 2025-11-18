#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const size_t DSIZE = 16384;      // matrix side dimension
const int block_size = 256;  // CUDA maximum is 1024
const int warpSize = 32;
// matrix row-sum kernel
__global__ void row_sums(const float *A, float *sums, size_t ds){

  int idx = threadIdx.x+blockDim.x*blockIdx.x; // create typical 1D thread index from built-in variables
  if (idx < ds){
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++)
      sum += A[idx*ds+i];         // write a for loop that will cause the thread to iterate across a row, keeeping a running sum, and write the result to sums
    sums[idx] = sum;
}}
// matrix column-sum kernel
__global__ void column_sums(const float *A, float *sums, size_t ds){

  int idx = threadIdx.x+blockDim.x*blockIdx.x; // create typical 1D thread index from built-in variables
  if (idx < ds){
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++)
      sum += A[idx+ds*i];         // write a for loop that will cause the thread to iterate down a column, keeeping a running sum, and write the result to sums
    sums[idx] = sum;
}}
bool validate(float *data, size_t sz){
  for (size_t i = 0; i < sz; i++)
    if (data[i] != (float)sz) {printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i], (float)sz); return false;}
    return true;
}


__global__ void row_sums_warp(const float *A, float *sums, size_t ds){
     float val = 0.0f;
     unsigned mask = 0xFFFFFFFFU;
     int lane = threadIdx.x % warpSize;
     int warpID = threadIdx.x / warpSize;
     
     int row_idx = blockIdx.x * (blockDim.x / warpSize) + warpID;
     if (row_idx < ds){
       for (size_t i = lane; i < ds; i += warpSize)
         val += A[row_idx*ds + i];  // each thread in the warp loads and sums multiple elements from the row
        __syncwarp();
      // first level of reduction within the warp
       for (int offset = warpSize/2; offset > 0; offset >>= 1) 
         val += __shfl_down_sync(mask, val, offset);
        if (lane == 0)
          sums[row_idx] = val;  // write the final sum for the row by lane 0 of the warp
     }
}

__global__ void row_sums_tb(const float *A, float *sums, size_t ds){
     float val = 0.0f;
     unsigned mask = 0xFFFFFFFFU;
     int lane = threadIdx.x % warpSize;
     int warpID = threadIdx.x / warpSize;
     int row = blockIdx.x;
     __shared__ float sdata[block_size/warpSize]; 

     if(row < ds)
     {
          // 网格块步进循环
          for(int offset = 0; offset < ds; offset += blockDim.x){
            int col = offset + threadIdx.x;
            if (col < ds)
                val += A[row*ds + col];  
          }
         __syncwarp();

          // warp 内归约
          for (int offset = warpSize/2; offset > 0; offset >>= 1) 
              val += __shfl_down_sync(mask, val, offset);
          if (lane == 0)
              sdata[warpID] = val;
          __syncthreads();

          // 线程块内归约
          if (warpID == 0){
              val = (lane < (blockDim.x/warpSize)) ? sdata[lane] : 0.0f;
              for (int offset = warpSize/2; offset > 0; offset >>= 1) 
                  val += __shfl_down_sync(mask, val, offset);
              if (lane == 0)
                  sums[row] = val;
          }
     }
}


int main(){

  float *h_A, *h_sums, *d_A, *d_sums;
  h_A = new float[DSIZE*DSIZE];  // allocate space for data in host memory
  h_sums = new float[DSIZE]();
  for (int i = 0; i < DSIZE*DSIZE; i++)  // initialize matrix in host memory
    h_A[i] = 1.0f;
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));  // allocate device space for A
  cudaMalloc(&d_sums, DSIZE*sizeof(float));  // allocate device space for vector d_sums
  cudaCheckErrors("cudaMalloc failure"); // error checking
  // copy matrix A to device:
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  //cuda processing sequence step 1 is complete
  row_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  if (!validate(h_sums, DSIZE)) return -1; 
  printf("row sums correct!\n");
  cudaMemset(d_sums, 0, DSIZE*sizeof(float));
  column_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  if (!validate(h_sums, DSIZE)) return -1; 
  printf("column sums correct!\n");

  cudaMemset(d_sums, 0, DSIZE*sizeof(float));
  row_sums_warp<<<(DSIZE+(block_size / warpSize)-1)/(block_size / warpSize), block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  if (!validate(h_sums, DSIZE)) return -1; 
  printf("Warp-level row sums correct!\n");

  cudaMemset(d_sums, 0, DSIZE*sizeof(float));
  row_sums_tb<<<DSIZE, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  if (!validate(h_sums, DSIZE)) return -1; 
  printf("Thread-block-level row sums correct!\n");
  return 0;
}
  
