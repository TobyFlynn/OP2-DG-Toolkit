#ifndef __DG_OP2_CUSTOM_BLAS_H
#define __DG_OP2_CUSTOM_BLAS_H

#include "op_seq.h"
#include "dg_mesh/dg_mesh.h"
#include "dg_constants/dg_constants.h"

bool op2_gemv_have_dp_custom_kernel(int m, int n);

bool op2_gemv_have_sp_custom_kernel(int m, int n);

#if !defined(OP2_DG_CUDA) && !defined(OP2_DG_HIP)
void op2_cpu_gemm(const int m, const int n, const int k,
                  const DG_FP alpha, const bool trans, const DG_FP *A,
                  const int lda, op_dat b_dat, const int ldb, const DG_FP beta,
                  op_dat c_dat, const int ldc);

void op2_cpu_gemm_halo_exchange(const int m, const int k,
                  const DG_FP alpha, const bool trans, const DG_FP *A,
                  const int lda, op_dat b_dat, const int ldb, const DG_FP beta,
                  op_dat c_dat, const int ldc);

void op2_cpu_gemm_sp(const int m, const int n, const int k,
                  const float alpha, const bool trans, const float *A_sp,
                  const int lda, op_dat b_dat, const int ldb, const float beta,
                  op_dat c_dat, const int ldc);

void op2_cpu_gemm_halo_exchange_sp(const int m, const int k,
                  const float alpha, const bool trans, const float *A_sp,
                  const int lda, op_dat b_dat, const int ldb, const float beta,
                  op_dat c_dat, const int ldc);
#else
void custom_kernel_gemv(op_set set, const bool t, const int m, const int n, const DG_FP alpha,
  const DG_FP beta, const DG_FP *matrix, op_dat arg4, op_dat arg5);
void custom_kernel_gemv_halo_exchange(op_set set, const bool t, const int m, const int n, const DG_FP alpha,
  const DG_FP beta, const DG_FP *matrix, op_dat arg4, op_dat arg5);
void custom_kernel_gemv_sp(op_set set, const bool t, const int m, const int n, const float alpha,
  const float beta, const float *matrix, op_dat arg4, op_dat arg5);
void custom_kernel_gemv_halo_exchange_sp(op_set set, const bool t, const int m, const int n, const float alpha,
  const float beta, const float *matrix, op_dat arg4, op_dat arg5);

void standard_blas_lib_gemv(op_set set, const bool t, const int m, const int n, const DG_FP alpha,
  const DG_FP beta, const DG_FP *matrix, op_dat arg4, op_dat arg5);
void standard_blas_lib_gemv_halo_exchange(op_set set, const bool t, const int m, const int n, const DG_FP alpha,
  const DG_FP beta, const DG_FP *matrix, op_dat arg4, op_dat arg5);
void standard_blas_lib_gemv_sp(op_set set, const bool t, const int m, const int n, const float alpha,
  const float beta, const float *matrix, op_dat arg4, op_dat arg5);
void standard_blas_lib_gemv_halo_exchange_sp(op_set set, const bool t, const int m, const int n, const float alpha,
  const float beta, const float *matrix, op_dat arg4, op_dat arg5);
#endif

#endif
