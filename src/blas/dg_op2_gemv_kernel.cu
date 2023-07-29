#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

#include "dg_compiler_defs.h"

#include "kernels/non_templated_aos.h"
#include "kernels/templated_aos.h"

#include <stdexcept>

cublasHandle_t cublas_handle;

void init_op2_gemv_cublas() {
  cublasCreate(&cublas_handle);
}

void destroy_op2_gemv_cublas() {
  cublasDestroy(cublas_handle);
}

void custom_kernel_gemv_sp(op_set set, const bool t, const int m, const int n, const float alpha,
  const float beta, const DG_FP *matrix, op_dat arg4, op_dat arg5) {
  throw std::runtime_error("SP");
}

void custom_kernel_gemv(op_set set, const bool t, const int m, const int n, const DG_FP alpha,
  const DG_FP beta, const DG_FP *matrix, op_dat x, op_dat y) {

  int nargs = 2;
  op_arg args[2] = {
    op_arg_dat(x, -1, OP_ID, x->dim, DG_FP_STR, OP_READ),
    op_arg_dat(y, -1, OP_ID, y->dim, DG_FP_STR, OP_RW)
  };

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2, 0);
  if (set_size > 0) {
    //set CUDA execution parameters
    int nthread = 256;

    if(t) {
      const int nblocks = set->size * n / nthread + 1;
      const int ncells = nthread / n + 3;
      cutilSafeCall(cudaFuncSetCacheConfig(templated_cuda_gemm_T_gpu<4>, cudaFuncCachePreferShared));
      cutilSafeCall(cudaFuncSetCacheConfig(templated_cuda_gemm_T_gpu<10>, cudaFuncCachePreferShared));
      cutilSafeCall(cudaFuncSetCacheConfig(templated_cuda_gemm_T_gpu<20>, cudaFuncCachePreferShared));
      cutilSafeCall(cudaFuncSetCacheConfig(templated_cuda_gemm_T_gpu<12>, cudaFuncCachePreferShared));
      cutilSafeCall(cudaFuncSetCacheConfig(templated_cuda_gemm_T_gpu<24>, cudaFuncCachePreferShared));
      cutilSafeCall(cudaFuncSetCacheConfig(templated_cuda_gemm_T_gpu<40>, cudaFuncCachePreferShared));
      cutilSafeCall(cudaFuncSetCacheConfig(cuda_gemm_T_gpu, cudaFuncCachePreferShared));
      switch(m) {
        // The number of nodes for each order
        case 4:
          templated_cuda_gemm_T_gpu<4><<<nblocks,nthread,(m*n + ncells * x->dim)*sizeof(double)>>>(n,
                                              x->dim, y->dim, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 10:
          templated_cuda_gemm_T_gpu<10><<<nblocks,nthread,(m*n + ncells * x->dim)*sizeof(double)>>>(n,
                                              x->dim, y->dim, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 20:
          templated_cuda_gemm_T_gpu<20><<<nblocks,nthread,(m*n + ncells * x->dim)*sizeof(double)>>>(n,
                                              x->dim, y->dim, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        // The number of face nodes for each order
        case 12:
          templated_cuda_gemm_T_gpu<12><<<nblocks,nthread,(m*n + ncells * x->dim)*sizeof(double)>>>(n,
                                              x->dim, y->dim, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 24:
          templated_cuda_gemm_T_gpu<24><<<nblocks,nthread,(m*n + ncells * x->dim)*sizeof(double)>>>(n,
                                              x->dim, y->dim, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 40:
          templated_cuda_gemm_T_gpu<40><<<nblocks,nthread,(m*n + ncells * x->dim)*sizeof(double)>>>(n,
                                              x->dim, y->dim, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        default:
          cuda_gemm_T_gpu<<<nblocks,nthread,(m*n + ncells * x->dim)*sizeof(double)>>>(m, n, x->dim, y->dim, alpha, beta,
                                               matrix, (double *) args[0].data_d,
                                               (double *) args[1].data_d, set->size);
      }
    } else {
      const int nblocks = set->size * m / nthread + 1;
      const int ncells = nthread / m + 3;
      cutilSafeCall(cudaFuncSetCacheConfig(templated_cuda_gemm_gpu<4>, cudaFuncCachePreferShared));
      cutilSafeCall(cudaFuncSetCacheConfig(templated_cuda_gemm_gpu<10>, cudaFuncCachePreferShared));
      cutilSafeCall(cudaFuncSetCacheConfig(templated_cuda_gemm_gpu<20>, cudaFuncCachePreferShared));
      cutilSafeCall(cudaFuncSetCacheConfig(templated_cuda_gemm_gpu<12>, cudaFuncCachePreferShared));
      cutilSafeCall(cudaFuncSetCacheConfig(templated_cuda_gemm_gpu<24>, cudaFuncCachePreferShared));
      cutilSafeCall(cudaFuncSetCacheConfig(templated_cuda_gemm_gpu<40>, cudaFuncCachePreferShared));
      cutilSafeCall(cudaFuncSetCacheConfig(cuda_gemm_gpu, cudaFuncCachePreferShared));
      switch(n) {
        // The number of nodes for each order
        case 4:
          templated_cuda_gemm_gpu<4><<<nblocks,nthread,(m*n + ncells * x->dim)*sizeof(double)>>>(m,
                                              x->dim, y->dim, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 10:
          templated_cuda_gemm_gpu<10><<<nblocks,nthread,(m*n + ncells * x->dim)*sizeof(double)>>>(m,
                                              x->dim, y->dim, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 20:
          templated_cuda_gemm_gpu<20><<<nblocks,nthread,(m*n + ncells * x->dim)*sizeof(double)>>>(m,
                                              x->dim, y->dim, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        // The number of face nodes for each order
        case 12:
          templated_cuda_gemm_gpu<12><<<nblocks,nthread,(m*n + ncells * x->dim)*sizeof(double)>>>(m,
                                              x->dim, y->dim, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 24:
          templated_cuda_gemm_gpu<24><<<nblocks,nthread,(m*n + ncells * x->dim)*sizeof(double)>>>(m,
                                              x->dim, y->dim, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 40:
          templated_cuda_gemm_gpu<40><<<nblocks,nthread,(m*n + ncells * x->dim)*sizeof(double)>>>(m,
                                              x->dim, y->dim, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        default:
          cuda_gemm_gpu<<<nblocks,nthread,(m*n + ncells * x->dim)*sizeof(double)>>>(m, n, x->dim, y->dim, alpha, beta,
                                             matrix, (double *) args[0].data_d,
                                             (double *) args[1].data_d, set->size);
      }
    }
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
}

void custom_kernel_gemv_halo_exchange(op_set set, const bool t, const int m, const int n, const DG_FP alpha,
  const DG_FP beta, const DG_FP *matrix, op_dat x, op_dat y) {

  throw std::runtime_error("gemv_halo_exchange not fully supported yet\n");
}
