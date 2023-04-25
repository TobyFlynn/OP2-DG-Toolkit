#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

#include "dg_compiler_defs.h"

// CUDA kernel function
__global__ void cuda_gemm_gpu(
  const int m, const int n, const int strideX, const int strideY,
  const double alpha, const double beta, const double * __restrict__ matrix,
  const double * __restrict__ arg4, double * __restrict__ arg5,
  const int set_size) {

  extern __shared__ double sh_mem[];
  double *mat_sh = sh_mem;
  double *x_sh = &sh_mem[m * n];

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = matrix[i];
  }
  __syncthreads();

  const int n_ = threadIdx.x + blockIdx.x * blockDim.x;
  if(n_ >= set_size) return;
  const int cell = n_;
  const double *x = arg4 + cell;
  double *y = arg5 + cell;
  for(int i = 0; i < m; i++) {
    DG_FP tmp = 0.0;
    for(int j = 0; j < n; j++) {
      int ind = DG_MAT_IND(i, j, m, n);
      tmp += mat_sh[ind] * x[j * strideX];
    }
    y[i * strideY] = beta * y[i * strideY] + alpha * tmp;
  }
}

__global__ void cuda_gemm_T_gpu(
  const int m, const int n, const int strideX, const int strideY,
  const double alpha, const double beta, const double *matrix,
  const double *__restrict arg4, double *arg5,
  int set_size) {

  extern __shared__ double sh_mem[];
  double *mat_sh = sh_mem;
  double *x_sh = &sh_mem[m * n];

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = matrix[i];
  }
  __syncthreads();

  const int n_ = threadIdx.x + blockIdx.x * blockDim.x;
  if(n_ >= set_size) return;
  const int cell = n_;
  const double *x = arg4 + cell;
  double *y = arg5 + cell;
  for(int i = 0; i < n; i++) {
    DG_FP tmp = 0.0;
    for(int j = 0; j < m; j++) {
      int ind = DG_MAT_IND(j, i, m, n);
      tmp += mat_sh[ind] * x[j * strideX];
    }
    y[i * strideY] = beta * y[i * strideY] + alpha * tmp;
  }
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
    const int nblocks = set->size / nthread + 1;
    const int strideX = getSetSizeFromOpArg(&args[0]);
    const int strideY = getSetSizeFromOpArg(&args[1]);

    if(t) {
      cuda_gemm_T_gpu<<<nblocks,nthread,m*n*sizeof(double)>>>(m, n, strideX, strideY, alpha, beta,
                                           matrix_d, (double *) args[0].data_d,
                                           (double *) args[1].data_d, set->size);
    } else {
      cuda_gemm_gpu<<<nblocks,nthread,m*n*sizeof(double)>>>(m, n, strideX, strideY, alpha, beta,
                                         matrix_d, (double *) args[0].data_d,
                                         (double *) args[1].data_d, set->size);
    }

    cudaFree(matrix_d);
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
}
