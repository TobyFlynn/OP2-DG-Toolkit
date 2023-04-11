#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

#include "dg_compiler_defs.h"

__device__ void _gemv_gpu(const int node, const int m, const int n,
                          const double alpha, const double beta,
                          const double *matrix, const double *x, double *y) {
  DG_FP tmp = 0.0;
  for(int j = 0; j < n; j++) {
    int ind = DG_MAT_IND(node, j, m, n);
    tmp += matrix[ind] * x[j];
  }
  y[node] = beta * y[node] + alpha * tmp;
}

__device__ void _gemv_T_gpu(const int node, const int m, const int n,
                            const double alpha, const double beta,
                            const double *matrix, const double *x, double *y) {
  DG_FP tmp = 0.0;
  for(int j = 0; j < m; j++) {
    int ind = DG_MAT_IND(j, node, m, n);
    tmp += matrix[ind] * x[j];
  }
  y[node] = beta * y[node] + alpha * tmp;
}

// CUDA kernel function
__global__ void _op_cuda_gemv(
  const int m, const int n, const int incX, const int incY,
  const double alpha, const double beta, const double *matrix,
  const double *__restrict arg4, double *arg5,
  int set_size) {


  //process set elements
  for(int n_ = threadIdx.x + blockIdx.x * blockDim.x;
      n_ < set_size * m; n_ += blockDim.x * gridDim.x){

    const int node = n_ % m;
    const int cell = n_ / m;
    _gemv_gpu(node, m, n, alpha, beta, matrix, arg4 + cell * incX,
              arg5 + cell * incY);
  }
}

__global__ void _op_cuda_gemv_T(
  const int m, const int n, const int incX, const int incY,
  const double alpha, const double beta, const double *matrix,
  const double *__restrict arg4, double *arg5,
  int set_size) {


  //process set elements
  for(int n_ = threadIdx.x + blockIdx.x * blockDim.x;
      n_ < set_size * m; n_ += blockDim.x * gridDim.x){

    const int node = n_ % n;
    const int cell = n_ / n;
    _gemv_T_gpu(node, m, n, alpha, beta, matrix, arg4 + cell * incX,
              arg5 + cell * incY);
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
    int nblocks = 200;

    if(t) {
      _op_cuda_gemv_T<<<nblocks,nthread>>>(m, n, x->dim, y->dim, alpha, beta,
                                           matrix_d, (double *) args[0].data_d,
                                           (double *) args[1].data_d, set->size);
    } else {
      _op_cuda_gemv<<<nblocks,nthread>>>(m, n, x->dim, y->dim, alpha, beta,
                                         matrix_d, (double *) args[0].data_d,
                                         (double *) args[1].data_d, set->size);
    }

    cudaFree(matrix_d);
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
}
