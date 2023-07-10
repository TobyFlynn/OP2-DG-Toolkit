#pragma once

template<int m, int n>
__global__ void templated_cuda_gemm_gpu_sp(const int strideX, const int strideY,
                      const float alpha, const float beta,
                      const float *matrix, const float * __restrict__ arg4,
                      float * __restrict__ arg5, const int set_size) {
  extern __shared__ float mat_sh_sp[];

  const int cell = threadIdx.x + blockIdx.x * blockDim.x;
  const float *x = arg4 + cell;
  float *y = arg5 + cell;
  float tmp_reg[m];

  if(threadIdx.x < m)
    mat_sh_sp[threadIdx.x] = matrix[DG_MAT_IND(threadIdx.x, 0, m, n)];
  __syncthreads();
  if(cell < set_size) {
    const float _x = x[0 * strideX];
    #pragma unroll
    for(int j = 0; j < m; j++) {
      tmp_reg[j] = mat_sh_sp[j] * _x;
    }
  }

  for(int i = 1; i < n; i++) {
    __syncthreads();
    if(threadIdx.x < m)
      mat_sh_sp[threadIdx.x] = matrix[DG_MAT_IND(threadIdx.x, i, m, n)];
    __syncthreads();
    if(cell < set_size) {
      const float _x = x[i * strideX];
      #pragma unroll
      for(int j = 0; j < m; j++) {
        tmp_reg[j] += mat_sh_sp[j] * _x;
      }
    }
  }

  if(cell < set_size) {
    #pragma unroll
    for(int i = 0; i < m; i++) {
      y[i * strideY] = beta * y[i * strideY] + alpha * tmp_reg[i];
    }
  }
}

template<int m>
__global__ void templated_cuda_gemm_T_gpu_sp(const int n, const int strideX,
                      const int strideY, const float alpha, const float beta,
                      const float *matrix, const float *__restrict__ arg4,
                      float *__restrict__ arg5, int set_size) {
  extern __shared__ float mat_sh_sp[];

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh_sp[i] = matrix[i];
  }
  __syncthreads();

  const int cell = threadIdx.x + blockIdx.x * blockDim.x;
  if(cell < set_size) {
    const float *x = arg4 + cell;
    float *y = arg5 + cell;
    float x_reg[m] = {};
    #pragma unroll
    for(int j = 0; j < m; j++) {
      x_reg[j] = x[j * strideX];
    }
    for(int i = 0; i < n; i++) {
      DG_FP tmp = 0.0;
      #pragma unroll
      for(int j = 0; j < m; j++) {
        int ind = DG_MAT_IND(j, i, m, n);
        tmp += mat_sh_sp[ind] * x_reg[j];
      }
      y[i * strideY] = beta * y[i * strideY] + alpha * tmp;
    }
  }
}
