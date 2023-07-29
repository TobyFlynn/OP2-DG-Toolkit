#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

#include "dg_compiler_defs.h"
#include "dg_mesh/dg_mesh.h"

#include "kernels/templated_soa.h"
#include "kernels/templated_soa_sp.h"

#include "cublas_v2.h"

cublasHandle_t cublas_handle;

void init_op2_gemv_cublas() {
  cublasCreate(&cublas_handle);
}

void destroy_op2_gemv_cublas() {
  cublasDestroy(cublas_handle);
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
                                            strideX, strideY, alpha,
                                            matrix, x_ptr, y_ptr, set_size);
    } else {
      templated_cuda_gemm_gpu_sp<m,n><<<nblocks,nthread,sh_mem_size>>>(
                                            strideX, strideY, alpha, beta,
                                            matrix, x_ptr, y_ptr, set_size);
    }
  }
}

void custom_kernel_gemv_sp(op_set set, const bool t, const int m, const int n, const float alpha,
  const float beta, const float *A_sp, op_dat x, op_dat y) {

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

    if(m == 4 && n == 4) {
      templated_wrapper_sp<4,4>(t, nblocks,nthread, n*m*sizeof(float),
                                strideX, strideY, alpha, beta, A_sp,
                                (float *) args[0].data_d,
                                (float *) args[1].data_d, set->size);
    } else if(m == 10 && n == 10) {
      templated_wrapper_sp<10,10>(t, nblocks,nthread, n*m*sizeof(float),
                                  strideX, strideY, alpha, beta, A_sp,
                                  (float *) args[0].data_d,
                                  (float *) args[1].data_d, set->size);
    } else if(m == 20 && n == 20) {
      templated_wrapper_sp<20,20>(t, nblocks,nthread, n*m*sizeof(float),
                                  strideX, strideY, alpha, beta, A_sp,
                                  (float *) args[0].data_d,
                                  (float *) args[1].data_d, set->size);
    } else if(m == 4 && n == 12) {
      templated_wrapper_sp<4,12>(t, nblocks,nthread, n*m*sizeof(float),
                                 strideX, strideY, alpha, beta, A_sp,
                                 (float *) args[0].data_d,
                                 (float *) args[1].data_d, set->size);
    } else if(m == 10 && n == 24) {
      templated_wrapper_sp<10,24>(t, nblocks,nthread, n*m*sizeof(float),
                                  strideX, strideY, alpha, beta, A_sp,
                                  (float *) args[0].data_d,
                                  (float *) args[1].data_d, set->size);
    } else if(m == 20 && n == 40) {
      templated_wrapper_sp<20,40>(t, nblocks,nthread, n*m*sizeof(float),
                                  strideX, strideY, alpha, beta, A_sp,
                                  (float *) args[0].data_d,
                                  (float *) args[1].data_d, set->size);
    } else if(m == 20 && n == 4) {
      templated_wrapper_sp<20,4>(t, nblocks,nthread, n*m*sizeof(float),
                                 strideX, strideY, alpha, beta, A_sp,
                                 (float *) args[0].data_d,
                                 (float *) args[1].data_d, set->size);
    } else if(m == 4 && n == 20) {
      templated_wrapper_sp<4,20>(t, nblocks,nthread, n*m*sizeof(float),
                                 strideX, strideY, alpha, beta, A_sp,
                                 (float *) args[0].data_d,
                                 (float *) args[1].data_d, set->size);
    } else {
      if(t) {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, set->size, n, m,
                      &alpha, (float *)args[0].data_d, strideX, A_sp, m,
                      &beta, (float *)args[1].data_d, strideY);
      } else {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, set->size, m, n,
                    &alpha, (float *)args[0].data_d, strideX, A_sp, m, &beta,
                    (float *)args[1].data_d, strideY);
      }
    }
  }

  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
}

void custom_kernel_gemv_halo_exchange_sp(op_set set, const bool t, const int m, const int n, const float alpha,
  const float beta, const float *A_sp, op_dat x, op_dat y) {

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

      if(m == 4 && n == 4) {
        templated_wrapper_sp<4,4>(t, nblocks,nthread, n*m*sizeof(float),
                                  strideX, strideY, alpha, beta, A_sp,
                                  x_ptr, y_ptr, end - start);
      } else if(m == 10 && n == 10) {
        templated_wrapper_sp<10,10>(t, nblocks,nthread, n*m*sizeof(float),
                                    strideX, strideY, alpha, beta, A_sp,
                                    x_ptr, y_ptr, end - start);
      } else if(m == 20 && n == 20) {
        templated_wrapper_sp<20,20>(t, nblocks,nthread, n*m*sizeof(float),
                                    strideX, strideY, alpha, beta, A_sp,
                                    x_ptr, y_ptr, end - start);
      } else if(m == 4 && n == 12) {
        templated_wrapper_sp<4,12>(t, nblocks,nthread, n*m*sizeof(float),
                                   strideX, strideY, alpha, beta, A_sp,
                                   x_ptr, y_ptr, end - start);
      } else if(m == 10 && n == 24) {
        templated_wrapper_sp<10,24>(t, nblocks,nthread, n*m*sizeof(float),
                                    strideX, strideY, alpha, beta, A_sp,
                                    x_ptr, y_ptr, end - start);
      } else if(m == 20 && n == 40) {
        templated_wrapper_sp<20,40>(t, nblocks,nthread, n*m*sizeof(float),
                                    strideX, strideY, alpha, beta, A_sp,
                                    x_ptr, y_ptr, end - start);
      } else if(m == 20 && n == 4) {
        templated_wrapper_sp<20,4>(t, nblocks,nthread, n*m*sizeof(float),
                                   strideX, strideY, alpha, beta, A_sp,
                                   x_ptr, y_ptr, end - start);
      } else if(m == 4 && n == 20) {
        templated_wrapper_sp<4,20>(t, nblocks,nthread, n*m*sizeof(float),
                                   strideX, strideY, alpha, beta, A_sp,
                                   x_ptr, y_ptr, end - start);
      } else {
        if(t) {
          cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, end - start, n,
                      m, &alpha, x_ptr, strideX, A_sp, m, &beta, y_ptr,
                      strideY);
        } else {
          cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, end - start, m,
                      n, &alpha, x_ptr, strideX, A_sp, m, &beta, y_ptr,
                      strideY);
        }
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
                                            strideX, strideY, alpha,
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

    if(m == 4 && n == 4) {
      templated_wrapper_dp<4,4>(t, nblocks,nthread, n*m*sizeof(double),
                                strideX, strideY, alpha, beta, matrix,
                                (double *) args[0].data_d,
                                (double *) args[1].data_d, set->size);
    } else if(m == 10 && n == 10) {
      templated_wrapper_dp<10,10>(t, nblocks,nthread, n*m*sizeof(double),
                                  strideX, strideY, alpha, beta, matrix,
                                  (double *) args[0].data_d,
                                  (double *) args[1].data_d, set->size);
    } else if(m == 20 && n == 20) {
      templated_wrapper_dp<20,20>(t, nblocks,nthread, n*m*sizeof(double),
                                  strideX, strideY, alpha, beta, matrix,
                                  (double *) args[0].data_d,
                                  (double *) args[1].data_d, set->size);
    } else if(m == 4 && n == 12) {
      templated_wrapper_dp<4,12>(t, nblocks,nthread, n*m*sizeof(double),
                                 strideX, strideY, alpha, beta, matrix,
                                 (double *) args[0].data_d,
                                 (double *) args[1].data_d, set->size);
    } else if(m == 10 && n == 24) {
      templated_wrapper_dp<10,24>(t, nblocks,nthread, n*m*sizeof(double),
                                  strideX, strideY, alpha, beta, matrix,
                                  (double *) args[0].data_d,
                                  (double *) args[1].data_d, set->size);
    } else if(m == 20 && n == 40) {
      templated_wrapper_dp<20,40>(t, nblocks,nthread, n*m*sizeof(double),
                                  strideX, strideY, alpha, beta, matrix,
                                  (double *) args[0].data_d,
                                  (double *) args[1].data_d, set->size);
    } else if(m == 20 && n == 4) {
      templated_wrapper_dp<20,4>(t, nblocks,nthread, n*m*sizeof(double),
                                 strideX, strideY, alpha, beta, matrix,
                                 (double *) args[0].data_d,
                                 (double *) args[1].data_d, set->size);
    } else if(m == 4 && n == 20) {
      templated_wrapper_dp<4,20>(t, nblocks,nthread, n*m*sizeof(double),
                                 strideX, strideY, alpha, beta, matrix,
                                 (double *) args[0].data_d,
                                 (double *) args[1].data_d, set->size);
    } else {
      if(t) {
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, set->size, n, m,
                      &alpha, (double *)args[0].data_d, strideX, matrix, m,
                      &beta, (double *)args[1].data_d, strideY);
      } else {
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, set->size, m, n,
                    &alpha, (double *)args[0].data_d, strideX, matrix, m, &beta,
                    (double *)args[1].data_d, strideY);
      }
    }
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
}

void custom_kernel_gemv_halo_exchange(op_set set, const bool t, const int m, const int n, const DG_FP alpha,
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

      if(m == 4 && n == 4) {
        templated_wrapper_dp<4,4>(t, nblocks,nthread, n*m*sizeof(double),
                                  strideX, strideY, alpha, beta, matrix,
                                  x_ptr, y_ptr, end - start);
      } else if(m == 10 && n == 10) {
        templated_wrapper_dp<10,10>(t, nblocks,nthread, n*m*sizeof(double),
                                    strideX, strideY, alpha, beta, matrix,
                                    x_ptr, y_ptr, end - start);
      } else if(m == 20 && n == 20) {
        templated_wrapper_dp<20,20>(t, nblocks,nthread, n*m*sizeof(double),
                                    strideX, strideY, alpha, beta, matrix,
                                    x_ptr, y_ptr, end - start);
      } else if(m == 4 && n == 12) {
        templated_wrapper_dp<4,12>(t, nblocks,nthread, n*m*sizeof(double),
                                   strideX, strideY, alpha, beta, matrix,
                                   x_ptr, y_ptr, end - start);
      } else if(m == 10 && n == 24) {
        templated_wrapper_dp<10,24>(t, nblocks,nthread, n*m*sizeof(double),
                                    strideX, strideY, alpha, beta, matrix,
                                    x_ptr, y_ptr, end - start);
      } else if(m == 20 && n == 40) {
        templated_wrapper_dp<20,40>(t, nblocks,nthread, n*m*sizeof(double),
                                    strideX, strideY, alpha, beta, matrix,
                                    x_ptr, y_ptr, end - start);
      } else if(m == 20 && n == 4) {
        templated_wrapper_dp<20,4>(t, nblocks,nthread, n*m*sizeof(double),
                                   strideX, strideY, alpha, beta, matrix,
                                   x_ptr, y_ptr, end - start);
      } else if(m == 4 && n == 20) {
        templated_wrapper_dp<4,20>(t, nblocks,nthread, n*m*sizeof(double),
                                   strideX, strideY, alpha, beta, matrix,
                                   x_ptr, y_ptr, end - start);
      } else {
        if(t) {
          cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, end - start, n,
                      m, &alpha, x_ptr, strideX, matrix, m, &beta, y_ptr,
                      strideY);
        } else {
          cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, end - start, m,
                      n, &alpha, x_ptr, strideX, matrix, m, &beta, y_ptr,
                      strideY);
        }
      }
    }
  }
  op_mpi_set_dirtybit_force_halo_exchange(nargs, args, 2);
  cutilSafeCall(cudaDeviceSynchronize());
}
