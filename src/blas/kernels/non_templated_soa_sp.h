#pragma once

__global__ void cuda_gemm_gpu_sp(
  const int m, const int n, const int strideX, const int strideY,
  const float alpha, const float beta, const float * __restrict__ matrix,
  const float * __restrict__ arg4, float * __restrict__ arg5,
  const int set_size) {

  extern __shared__ float sh_mem_sp[];
  float *mat_sh = sh_mem_sp;
  float *x_sh = &sh_mem_sp[m * n];

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = matrix[i];
  }
  __syncthreads();

  const int cell = threadIdx.x + blockIdx.x * blockDim.x;
  if(cell >= set_size) return;
  const float *x = arg4 + cell;
  float *y = arg5 + cell;
  for(int i = 0; i < m; i++) {
    DG_FP tmp = 0.0;
    for(int j = 0; j < n; j++) {
      int ind = DG_MAT_IND(i, j, m, n);
      tmp += mat_sh[ind] * x[j * strideX];
    }
    y[i * strideY] = beta * y[i * strideY] + alpha * tmp;
  }
}

__global__ void cuda_gemm_T_gpu_sp(
  const int m, const int n, const int strideX, const int strideY,
  const float alpha, const float beta, const float *matrix,
  const float *__restrict arg4, float *arg5,
  int set_size) {

  extern __shared__ float sh_mem_sp[];
  float *mat_sh = sh_mem_sp;
  float *x_sh = &sh_mem_sp[m * n];

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = matrix[i];
  }
  __syncthreads();

  const int cell = threadIdx.x + blockIdx.x * blockDim.x;
  if(cell >= set_size) return;
  const float *x = arg4 + cell;
  float *y = arg5 + cell;
  for(int i = 0; i < n; i++) {
    DG_FP tmp = 0.0;
    for(int j = 0; j < m; j++) {
      int ind = DG_MAT_IND(j, i, m, n);
      tmp += mat_sh[ind] * x[j * strideX];
    }
    y[i * strideY] = beta * y[i * strideY] + alpha * tmp;
  }
}
