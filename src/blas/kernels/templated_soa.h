#pragma once

template<int m, int n>
__global__ void templated_cuda_gemm_gpu(const int strideX, const int strideY,
                      const double alpha, const double beta,
                      const double *matrix, const double * __restrict__ arg4,
                      double * __restrict__ arg5, const int set_size) {
  extern __shared__ double mat_sh[];

  const int cell = threadIdx.x + blockIdx.x * blockDim.x;
  const double *x = arg4 + cell;
  double *y = arg5 + cell;
  double tmp_reg[m];

  if(threadIdx.x < m)
    mat_sh[threadIdx.x] = matrix[DG_MAT_IND(threadIdx.x, 0, m, n)];
  __syncthreads();
  if(cell < set_size) {
    const double _x = x[0 * strideX];
    #pragma unroll
    for(int j = 0; j < m; j++) {
      tmp_reg[j] = mat_sh[j] * _x;
    }
  }

  for(int i = 1; i < n; i++) {
    __syncthreads();
    if(threadIdx.x < m)
      mat_sh[threadIdx.x] = matrix[DG_MAT_IND(threadIdx.x, i, m, n)];
    __syncthreads();
    if(cell < set_size) {
      const double _x = x[i * strideX];
      #pragma unroll
      for(int j = 0; j < m; j++) {
        tmp_reg[j] += mat_sh[j] * _x;
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
__global__ void templated_cuda_gemm_T_gpu(const int n, const int strideX,
                      const int strideY, const double alpha, const double beta,
                      const double *matrix, const double *__restrict__ arg4,
                      double *__restrict__ arg5, int set_size) {
  extern __shared__ double mat_sh[];

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = matrix[i];
  }
  __syncthreads();

  const int cell = threadIdx.x + blockIdx.x * blockDim.x;
  if(cell < set_size) {
    const double *x = arg4 + cell;
    double *y = arg5 + cell;
    double x_reg[m] = {};
    #pragma unroll
    for(int j = 0; j < m; j++) {
      x_reg[j] = x[j * strideX];
    }
    for(int i = 0; i < n; i++) {
      DG_FP tmp = 0.0;
      #pragma unroll
      for(int j = 0; j < m; j++) {
        int ind = DG_MAT_IND(j, i, m, n);
        tmp += mat_sh[ind] * x_reg[j];
      }
      y[i * strideY] = beta * y[i * strideY] + alpha * tmp;
    }
  }
}
