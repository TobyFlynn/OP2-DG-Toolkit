#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

#include "dg_compiler_defs.h"
#include "dg_mesh/dg_mesh.h"
#include "dg_op2_custom_blas.h"

#include "kernels/templated_soa.h"
#include "kernels/templated_soa_sp.h"

#include "cublas_v2.h"

cublasHandle_t cublas_handle;

void init_op2_gemv() {
  cublasCreate(&cublas_handle);
}

void destroy_op2_gemv() {
  cublasDestroy(cublas_handle);
}

bool op2_gemv_have_dp_custom_kernel(int m, int n) {
  [OP2_DG_GPU_SOA_HAVE_DP_CUSTOM_KERNEL]
  else {
    return false;
  }
}

bool op2_gemv_have_sp_custom_kernel(int m, int n) {
  [OP2_DG_GPU_SOA_HAVE_SP_CUSTOM_KERNEL]
  else {
    return false;
  }
}

template<int m, int n>
void templated_wrapper_sp(bool trans, int nblocks, int nthread, int sh_mem_size,
                          const int strideX, const int strideY,
                          const float alpha, const float beta,
                          const float * __restrict__ matrix,
                          const float * __restrict__ x_ptr,
                          float * __restrict__ y_ptr, const int set_size) {
  if(trans) {
    if(beta == 0.0) {
      if(alpha == 1.0) {
        templated_cuda_gemm_T_gpu_1_alpha_0_beta_sp<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                              strideX, strideY,
                                              matrix, x_ptr, y_ptr, set_size);
      } else {
        templated_cuda_gemm_T_gpu_0_beta_sp<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                              strideX, strideY, alpha,
                                              matrix, x_ptr, y_ptr, set_size);
      }
    } else if(beta == 1.0) {
      if(alpha == 1.0) {
        templated_cuda_gemm_T_gpu_1_alpha_1_beta_sp<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                              strideX, strideY,
                                              matrix, x_ptr, y_ptr, set_size);
      } else {
        templated_cuda_gemm_T_gpu_1_beta_sp<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                              strideX, strideY, alpha,
                                              matrix, x_ptr, y_ptr, set_size);
      }
    } else if (alpha == 1.0) {
      templated_cuda_gemm_T_gpu_1_alpha_sp<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                            strideX, strideY, beta,
                                            matrix, x_ptr, y_ptr, set_size);
    } else {
      templated_cuda_gemm_T_gpu_sp<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                            strideX, strideY, alpha, beta,
                                            matrix, x_ptr, y_ptr, set_size);
    }
  } else {
    if(beta == 0.0) {
      if(alpha == 1.0) {
        templated_cuda_gemm_gpu_1_alpha_0_beta_sp<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                              strideX, strideY,
                                              matrix, x_ptr, y_ptr, set_size);
      } else {
        templated_cuda_gemm_gpu_0_beta_sp<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                              strideX, strideY, alpha,
                                              matrix, x_ptr, y_ptr, set_size);
      }
    } else if(beta == 1.0) {
      if(alpha == 1.0) {
        templated_cuda_gemm_gpu_1_alpha_1_beta_sp<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                              strideX, strideY,
                                              matrix, x_ptr, y_ptr, set_size);
      } else {
        templated_cuda_gemm_gpu_1_beta_sp<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                              strideX, strideY, alpha,
                                              matrix, x_ptr, y_ptr, set_size);
      }
    } else if (alpha == 1.0) {
      templated_cuda_gemm_gpu_1_alpha_sp<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                            strideX, strideY, beta,
                                            matrix, x_ptr, y_ptr, set_size);
    } else {
      templated_cuda_gemm_gpu_sp<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                            strideX, strideY, alpha, beta,
                                            matrix, x_ptr, y_ptr, set_size);
    }
  }
}

void custom_kernel_gemv_sp(op_set set, const bool t, const int m, const int n, const float alpha,
  const float beta, const float *matrix, op_dat x, op_dat y) {

  if(!op2_gemv_have_sp_custom_kernel(m, n)) {
    standard_blas_lib_gemv_sp(set, t, m, n, alpha, beta, matrix, x, y);
    return;
  }

  int nargs = 2;
  op_arg args[2] = {
    op_arg_dat(x, -1, OP_ID, x->dim, "float", OP_READ),
    op_arg_dat(y, -1, OP_ID, y->dim, "float", OP_RW)
  };

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2, 0);
  if (set_size > 0) {
    //set CUDA execution parameters
    int nthread = 256;
    const int nblocks = (set->size - 1) / nthread + 1;
    const int strideX = getSetSizeFromOpArg(&args[0]);
    const int strideY = getSetSizeFromOpArg(&args[1]);
    const float *x_ptr = (float *)args[0].data_d;
    float *y_ptr = (float *)args[1].data_d;
    const int num_vecs = set->size;

    [OP2_DG_GPU_SOA_SP_BLAS_STUB]

  }

  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
}

void standard_blas_lib_gemv_sp(op_set set, const bool t, const int m, const int n, const float alpha,
  const float beta, const float *matrix, op_dat x, op_dat y) {
  int nargs = 2;
  op_arg args[2] = {
    op_arg_dat(x, -1, OP_ID, x->dim, "float", OP_READ),
    op_arg_dat(y, -1, OP_ID, y->dim, "float", OP_RW)
  };

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2, 0);
  if (set_size > 0) {
    //set CUDA execution parameters
    int nthread = 256;
    const int nblocks = (set->size - 1) / nthread + 1;
    const int strideX = getSetSizeFromOpArg(&args[0]);
    const int strideY = getSetSizeFromOpArg(&args[1]);
    const float *x_ptr = (float *)args[0].data_d;
    float *y_ptr = (float *)args[1].data_d;
    const int num_vecs = set->size;

    if(t) {
      cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, num_vecs, n, m,
                  &alpha, x_ptr, strideX, matrix, m, &beta, y_ptr, strideY);
    } else {
      cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, num_vecs, m, n,
                  &alpha, x_ptr, strideX, matrix, m, &beta, y_ptr, strideY);
    }
  }

  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
}

void custom_kernel_gemv_halo_exchange_sp(op_set set, const bool t, const int m, const int n, const float alpha,
  const float beta, const float *matrix, op_dat x, op_dat y) {

  if(!op2_gemv_have_sp_custom_kernel(m, n)) {
    standard_blas_lib_gemv_halo_exchange_sp(set, t, m, n, alpha, beta, matrix, x, y);
    return;
  }

  int nargs = 2;
  op_arg args[2] = {
    op_arg_dat(x, -1, OP_ID, x->dim, "float", OP_READ),
    op_arg_dat(y, -1, OP_ID, y->dim, "float", beta == 0.0 ? OP_WRITE : OP_RW)
  };

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2, 1);

  if (set_size > 0) {

    const int strideX = getSetSizeFromOpArg(&args[0]);
    const int strideY = getSetSizeFromOpArg(&args[1]);

    for ( int round=0; round<2; round++ ){
      if (round==1) {
        op_mpi_wait_all_grouped(nargs, args, 2, 1);
      }

      int start = round==0 ? 0 : set->size;
      int end = round==0 ? set->size : set->size + set->exec_size + set->nonexec_size;
      if(end - start <= 0) continue;

      const float *x_ptr = (float *)args[0].data_d + start;
      float *y_ptr = (float *)args[1].data_d + start;
      const int nthread = 256;
      const int nblocks = (end - start) / nthread + 1;
      const int num_vecs = end - start;

      [OP2_DG_GPU_SOA_SP_BLAS_STUB]

    }
  }

  op_mpi_set_dirtybit_force_halo_exchange(nargs, args, 2);
  cutilSafeCall(cudaDeviceSynchronize());
}

void standard_blas_lib_gemv_halo_exchange_sp(op_set set, const bool t, const int m, const int n, const float alpha,
  const float beta, const float *matrix, op_dat x, op_dat y) {

  int nargs = 2;
  op_arg args[2] = {
    op_arg_dat(x, -1, OP_ID, x->dim, "float", OP_READ),
    op_arg_dat(y, -1, OP_ID, y->dim, "float", beta == 0.0 ? OP_WRITE : OP_RW)
  };

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2, 1);

  if (set_size > 0) {

    const int strideX = getSetSizeFromOpArg(&args[0]);
    const int strideY = getSetSizeFromOpArg(&args[1]);

    for ( int round=0; round<2; round++ ){
      if (round==1) {
        op_mpi_wait_all_grouped(nargs, args, 2, 1);
      }

      int start = round==0 ? 0 : set->size;
      int end = round==0 ? set->size : set->size + set->exec_size + set->nonexec_size;
      if(end - start <= 0) continue;

      const float *x_ptr = (float *)args[0].data_d + start;
      float *y_ptr = (float *)args[1].data_d + start;
      const int nthread = 256;
      const int nblocks = (end - start) / nthread + 1;
      const int num_vecs = end - start;

      if(t) {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, num_vecs, n,
                    m, &alpha, x_ptr, strideX, matrix, m, &beta, y_ptr, strideY);
      } else {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, num_vecs, m,
                    n, &alpha, x_ptr, strideX, matrix, m, &beta, y_ptr, strideY);
      }
    }
  }

  op_mpi_set_dirtybit_force_halo_exchange(nargs, args, 2);
  cutilSafeCall(cudaDeviceSynchronize());
}

template<int m, int n>
void templated_wrapper_dp(bool trans, int nblocks, int nthread, int sh_mem_size,
                          const int strideX, const int strideY,
                          const double alpha, const double beta,
                          const double * __restrict__ matrix,
                          const double * __restrict__ x_ptr,
                          double * __restrict__ y_ptr, const int set_size) {
  if(trans) {
    if(beta == 0.0) {
      if(alpha == 1.0) {
        templated_cuda_gemm_T_gpu_1_alpha_0_beta<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                              strideX, strideY,
                                              matrix, x_ptr, y_ptr, set_size);
      } else {
        templated_cuda_gemm_T_gpu_0_beta<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                              strideX, strideY, alpha,
                                              matrix, x_ptr, y_ptr, set_size);
      }
    } else if(beta == 1.0) {
      if(alpha == 1.0) {
        templated_cuda_gemm_T_gpu_1_alpha_1_beta<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                              strideX, strideY,
                                              matrix, x_ptr, y_ptr, set_size);
      } else {
        templated_cuda_gemm_T_gpu_1_beta<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                              strideX, strideY, alpha,
                                              matrix, x_ptr, y_ptr, set_size);
      }
    } else if (alpha == 1.0) {
      templated_cuda_gemm_T_gpu_1_alpha<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                            strideX, strideY, beta,
                                            matrix, x_ptr, y_ptr, set_size);
    } else {
      templated_cuda_gemm_T_gpu<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                            strideX, strideY, alpha, beta,
                                            matrix, x_ptr, y_ptr, set_size);
    }
  } else {
    if(beta == 0.0) {
      if(alpha == 1.0) {
        templated_cuda_gemm_gpu_1_alpha_0_beta<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                              strideX, strideY,
                                              matrix, x_ptr, y_ptr, set_size);
      } else {
        templated_cuda_gemm_gpu_0_beta<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                              strideX, strideY, alpha,
                                              matrix, x_ptr, y_ptr, set_size);
      }
    } else if(beta == 1.0) {
      if(alpha == 1.0) {
        templated_cuda_gemm_gpu_1_alpha_1_beta<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                              strideX, strideY,
                                              matrix, x_ptr, y_ptr, set_size);
      } else {
        templated_cuda_gemm_gpu_1_beta<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                              strideX, strideY, alpha,
                                              matrix, x_ptr, y_ptr, set_size);
      }
    } else if (alpha == 1.0) {
      templated_cuda_gemm_gpu_1_alpha<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                            strideX, strideY, beta,
                                            matrix, x_ptr, y_ptr, set_size);
    } else {
      templated_cuda_gemm_gpu<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                            strideX, strideY, alpha, beta,
                                            matrix, x_ptr, y_ptr, set_size);
    }
  }
}

void custom_kernel_gemv(op_set set, const bool t, const int m, const int n, const DG_FP alpha,
  const DG_FP beta, const DG_FP *matrix, op_dat x, op_dat y) {

  if(!op2_gemv_have_dp_custom_kernel(m, n)) {
    standard_blas_lib_gemv(set, t, m, n, alpha, beta, matrix, x, y);
    return;
  }

  int nargs = 2;
  op_arg args[2] = {
    op_arg_dat(x, -1, OP_ID, x->dim, DG_FP_STR, OP_READ),
    op_arg_dat(y, -1, OP_ID, y->dim, DG_FP_STR, OP_RW)
  };

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2, 0);
  if (set_size > 0) {
    //set CUDA execution parameters
    int nthread = 256;
    const int nblocks = (set->size - 1) / nthread + 1;
    const int strideX = getSetSizeFromOpArg(&args[0]);
    const int strideY = getSetSizeFromOpArg(&args[1]);
    const double *x_ptr = (double *)args[0].data_d;
    double *y_ptr = (double *)args[1].data_d;
    const int num_vecs = set->size;

    [OP2_DG_GPU_SOA_DP_BLAS_STUB]

  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
}

void standard_blas_lib_gemv(op_set set, const bool t, const int m, const int n, const DG_FP alpha,
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
    const int nblocks = (set->size - 1) / nthread + 1;
    const int strideX = getSetSizeFromOpArg(&args[0]);
    const int strideY = getSetSizeFromOpArg(&args[1]);
    const double *x_ptr = (double *)args[0].data_d;
    double *y_ptr = (double *)args[1].data_d;
    const int num_vecs = set->size;

    if(t) {
      cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, num_vecs, n, m,
                  &alpha, x_ptr, strideX, matrix, m, &beta, y_ptr, strideY);
    } else {
      cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, num_vecs, m, n,
                  &alpha, x_ptr, strideX, matrix, m, &beta, y_ptr, strideY);
    }
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
}

void custom_kernel_gemv_halo_exchange(op_set set, const bool t, const int m, const int n, const DG_FP alpha,
  const DG_FP beta, const DG_FP *matrix, op_dat x, op_dat y) {

  if(!op2_gemv_have_dp_custom_kernel(m, n)) {
    standard_blas_lib_gemv_halo_exchange(set, t, m, n, alpha, beta, matrix, x, y);
    return;
  }

  int nargs = 2;
  op_arg args[2] = {
    op_arg_dat(x, -1, OP_ID, x->dim, DG_FP_STR, OP_READ),
    op_arg_dat(y, -1, OP_ID, y->dim, DG_FP_STR, beta == 0.0 ? OP_WRITE : OP_RW)
  };

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2, 1);
  if (set_size > 0) {
    //set CUDA execution parameters
    const int strideX = getSetSizeFromOpArg(&args[0]);
    const int strideY = getSetSizeFromOpArg(&args[1]);

    for ( int round=0; round<2; round++ ){
      if (round==1) {
        op_mpi_wait_all_grouped(nargs, args, 2, 1);
      }

      int start = round==0 ? 0 : set->size;
      int end = round==0 ? set->size : set->size + set->exec_size + set->nonexec_size;
      if(end - start <= 0) continue;

      const double *x_ptr = (double *)args[0].data_d + start;
      double *y_ptr = (double *)args[1].data_d + start;
      const int nthread = 256;
      const int nblocks = (end - start) / nthread + 1;
      const int num_vecs = end - start;

      [OP2_DG_GPU_SOA_DP_BLAS_STUB]

    }
  }
  op_mpi_set_dirtybit_force_halo_exchange(nargs, args, 2);
  cutilSafeCall(cudaDeviceSynchronize());
}

void standard_blas_lib_gemv_halo_exchange(op_set set, const bool t, const int m, const int n, const DG_FP alpha,
  const DG_FP beta, const DG_FP *matrix, op_dat x, op_dat y) {

  int nargs = 2;
  op_arg args[2] = {
    op_arg_dat(x, -1, OP_ID, x->dim, DG_FP_STR, OP_READ),
    op_arg_dat(y, -1, OP_ID, y->dim, DG_FP_STR, beta == 0.0 ? OP_WRITE : OP_RW)
  };

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2, 1);
  if (set_size > 0) {
    //set CUDA execution parameters
    const int strideX = getSetSizeFromOpArg(&args[0]);
    const int strideY = getSetSizeFromOpArg(&args[1]);

    for ( int round=0; round<2; round++ ){
      if (round==1) {
        op_mpi_wait_all_grouped(nargs, args, 2, 1);
      }

      int start = round==0 ? 0 : set->size;
      int end = round==0 ? set->size : set->size + set->exec_size + set->nonexec_size;
      if(end - start <= 0) continue;

      const double *x_ptr = (double *)args[0].data_d + start;
      double *y_ptr = (double *)args[1].data_d + start;
      const int nthread = 256;
      const int nblocks = (end - start) / nthread + 1;
      const int num_vecs = end - start;

      if(t) {
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, num_vecs, n,
                    m, &alpha, x_ptr, strideX, matrix, m, &beta, y_ptr, strideY);
      } else {
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, num_vecs, m,
                    n, &alpha, x_ptr, strideX, matrix, m, &beta, y_ptr, strideY);
      }

    }
  }
  op_mpi_set_dirtybit_force_halo_exchange(nargs, args, 2);
  cutilSafeCall(cudaDeviceSynchronize());
}
