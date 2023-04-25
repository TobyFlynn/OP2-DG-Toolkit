#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

#include "dg_compiler_defs.h"

// CUDA kernel function
__global__ void cuda_gemm_gpu(
  const int m, const int n, const int incX, const int incY,
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
  for(int j = 0; j < n; j++) {
    int ind = DG_MAT_IND(node, j, m, n);
    tmp += mat_sh[ind] * x[j];
  }
  y[node] = beta * y[node] + alpha * tmp;
}

__global__ void cuda_gemm_T_gpu(
  const int m, const int n, const int incX, const int incY,
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
  for(int j = 0; j < m; j++) {
    int ind = DG_MAT_IND(j, node, m, n);
    tmp += mat_sh[ind] * x[j];
  }
  y[node] = beta * y[node] + alpha * tmp;
}


//host stub function
void custom_kernel_gemv(op_set set, const bool t, const int m, const int n, const DG_FP alpha,
  const DG_FP beta, const DG_FP *matrix, op_dat x, op_dat y) {

  int nargs = 2;
  op_arg args[2] = {
    op_arg_dat(x, -1, OP_ID, x->dim, DG_FP_STR, OP_READ),
    op_arg_dat(y, -1, OP_ID, y->dim, DG_FP_STR, OP_WRITE)
  };

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2);
  if (set_size > 0) {
    DG_FP *matrix_d;
    cudaMalloc(&matrix_d, m * n * sizeof(DG_FP));
    cudaMemcpy(matrix_d, matrix, m * n * sizeof(DG_FP), cudaMemcpyHostToDevice);

    //set CUDA execution parameters
    int nthread = 256;

    if(t) {
      const int nblocks = set->size * n / nthread + 1;
      const int ncells = nthread / n + 3;
      cuda_gemm_T_gpu<<<nblocks,nthread,(m*n + ncells * x->dim)*sizeof(double)>>>(m, n, x->dim, y->dim, alpha, beta,
                                           matrix_d, (double *) args[0].data_d,
                                           (double *) args[1].data_d, set->size);
    } else {
      const int nblocks = set->size * m / nthread + 1;
      const int ncells = nthread / m + 3;
      cuda_gemm_gpu<<<nblocks,nthread,(m*n + ncells * x->dim)*sizeof(double)>>>(m, n, x->dim, y->dim, alpha, beta,
                                         matrix_d, (double *) args[0].data_d,
                                         (double *) args[1].data_d, set->size);
    }

    cudaFree(matrix_d);
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
}
