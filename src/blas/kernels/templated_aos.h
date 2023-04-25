#pragma once

template<int n>
__global__ void templated_cuda_gemm_gpu(
  const int m, const int incX, const int incY,
  const double alpha, const double beta, const double * __restrict__ matrix,
  const double * __restrict__ arg4, double * __restrict__ arg5,
  int set_size) {

  extern __shared__ double sh_mem[];
  double *mat_sh = sh_mem;
  double *x_sh = &sh_mem[m * n];

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = matrix[i];
  }

  const int n_ = threadIdx.x + blockIdx.x * blockDim.x;
  const int start_n = n_ - threadIdx.x;
  const int num_cells = min((start_n + blockDim.x) / m, set_size) - (start_n / m) + 1;
  const int start_ind = (start_n / m) * incX;
  const int num_load = num_cells * incX;
  for(int i = threadIdx.x; i < num_load; i += blockDim.x) {
    x_sh[i] = arg4[start_ind + i];
  }
  const int local_cell_id = (n_ / m) - (start_n / m);
  __syncthreads();

  if(n_ >= set_size * m) return;
  const int node = n_ % m;
  const int cell = n_ / m;
  const double *x = x_sh + local_cell_id * incX;
  double *y = arg5 + cell * incY;
  DG_FP tmp = 0.0;
  #pragma unroll
  for(int j = 0; j < n; j++) {
    int ind = DG_MAT_IND(node, j, m, n);
    tmp += mat_sh[ind] * x[j];
  }
  y[node] = beta * y[node] + alpha * tmp;
}

template<int m>
__global__ void templated_cuda_gemm_T_gpu(
  const int n, const int incX, const int incY,
  const double alpha, const double beta, const double *matrix,
  const double *__restrict arg4, double *arg5,
  int set_size) {

  extern __shared__ double sh_mem[];
  double *mat_sh = sh_mem;
  double *x_sh = &sh_mem[m * n];

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = matrix[i];
  }

  const int n_ = threadIdx.x + blockIdx.x * blockDim.x;
  const int start_n = n_ - threadIdx.x;
  const int num_cells = min((start_n + blockDim.x) / n, set_size) - (start_n / n) + 1;
  const int start_ind = (start_n / n) * incX;
  const int num_load = num_cells * incX;
  for(int i = threadIdx.x; i < num_load; i += blockDim.x) {
    x_sh[i] = arg4[start_ind + i];
  }
  const int local_cell_id = (n_ / n) - (start_n / n);
  __syncthreads();

  if(n_ >= set_size * n) return;
  const int node = n_ % n;
  const int cell = n_ / n;
  const double *x = x_sh + local_cell_id * incX;
  double *y = arg5 + cell * incY;
  DG_FP tmp = 0.0;
  #pragma unroll
  for(int j = 0; j < m; j++) {
    int ind = DG_MAT_IND(j, node, m, n);
    tmp += mat_sh[ind] * x[j];
  }
  y[node] = beta * y[node] + alpha * tmp;
}
