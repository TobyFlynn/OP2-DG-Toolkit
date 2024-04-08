#include "dg_op2_blas.h"

#include "op_seq.h"

#include "dg_compiler_defs.h"
#include "dg_abort.h"

#include <iostream>

extern DGConstants *constants;

#if DG_DIM == 2
#include "dg_global_constants/dg_global_constants_2d.h"
#else
#include "dg_global_constants/dg_global_constants_3d.h"
#include "dg_mesh/dg_mesh_3d.h"
#endif

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
#endif

void op2_gemv(DGMesh *mesh, bool transpose, const DG_FP alpha,
              DGConstants::Constant_Matrix matrix, op_dat x, const DG_FP beta,
              op_dat y) {
  // Get the DGConstantMatrix
  DGConstantMatrix *matrix_ptr = constants->get_dg_constant_matrix_ptr(matrix);
  // Get dimensions of the matrix
  const int rows = matrix_ptr->get_rows(mesh->order_int);
  const int cols = matrix_ptr->get_cols(mesh->order_int);
  // Check that both op_dats are defined on the same set
  if(x->set->index != y->set->index)
    dg_abort("op_dats passed to op2_gemv are defined on different op_sets");
  const int n = x->set->size;

  #if defined(OP2_DG_CUDA) || defined(OP2_DG_HIP)
  custom_kernel_gemv(x->set, transpose, rows, cols, alpha, beta, matrix_ptr->get_mat_ptr_dp_device(mesh->order_int), x, y);
  #else
  op2_cpu_gemm(transpose ? cols : rows, n, transpose ? rows : cols, alpha, transpose, matrix_ptr->get_mat_ptr_dp_device(mesh->order_int), rows, x, x->dim, beta, y, y->dim);
  #endif
}

void op2_gemv_interp(DGMesh *mesh, const int from_N, const int to_N, op_dat x, op_dat y) {
  if(from_N == to_N) {
    op_par_loop(copy_dg_np_tk, "copy_dg_np_tk", mesh->cells,
                op_arg_dat(x, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));
    return;
  }

  const int from_NP = DG_CONSTANTS_TK[(from_N - 1) * DG_NUM_CONSTANTS];
  const int to_NP = DG_CONSTANTS_TK[(to_N - 1) * DG_NUM_CONSTANTS];
  const int m = to_NP;
  const int n = mesh->cells->size;
  const int k = from_NP;
  const DG_FP *A = constants->get_mat_ptr_device(DGConstants::INTERP_MATRIX_ARRAY) + ((from_N - 1) * DG_ORDER + (to_N - 1)) * DG_NP * DG_NP;

  // TODO 2D
  #if defined(USE_OP2_KERNELS)
  op_par_loop(interp_dat_to_new_order_3d_copy, "interp_dat_to_new_order_3d_copy", mesh->cells,
              op_arg_gbl(constants->get_mat_ptr_device(DGConstants::INTERP_MATRIX_ARRAY), DG_ORDER * DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(&from_N, 1, "int", OP_READ),
              op_arg_gbl(&to_N, 1, "int", OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));
  #elif defined(OP2_DG_CUDA) || defined(OP2_DG_HIP)
  custom_kernel_gemv(mesh->cells, false, m, k, 1.0, 0.0, A, x, y);
  #else
  op2_cpu_gemm(m, n, k, 1.0, false, A, m, x, DG_NP, 0.0, y, DG_NP);
  #endif
}

void op2_gemv_halo_exchange(DGMesh *mesh, bool transpose, const DG_FP alpha,
              DGConstants::Constant_Matrix matrix, op_dat x, const DG_FP beta,
              op_dat y) {
  // Get the DGConstantMatrix
  DGConstantMatrix *matrix_ptr = constants->get_dg_constant_matrix_ptr(matrix);
  // Get dimensions of the matrix
  const int rows = matrix_ptr->get_rows(mesh->order_int);
  const int cols = matrix_ptr->get_cols(mesh->order_int);
  // Check that both op_dats are defined on the same set
  if(x->set->index != y->set->index)
    dg_abort("op_dats passed to op2_gemv_halo_exchange are defined on different op_sets");

  #if defined(OP2_DG_CUDA) || defined(OP2_DG_HIP)
  custom_kernel_gemv_halo_exchange(x->set, transpose, rows, cols, alpha, beta, matrix_ptr->get_mat_ptr_dp_device(mesh->order_int), x, y);
  #else
  op2_cpu_gemm_halo_exchange(transpose ? cols : rows, transpose ? rows : cols, alpha, transpose, matrix_ptr->get_mat_ptr_dp_device(mesh->order_int), rows, x, x->dim, beta, y, y->dim);
  #endif
}

void op2_gemv_sp(DGMesh *mesh, bool transpose, const DG_FP alpha,
              DGConstants::Constant_Matrix matrix, op_dat x, const DG_FP beta,
              op_dat y) {
  // Get the DGConstantMatrix
  DGConstantMatrix *matrix_ptr = constants->get_dg_constant_matrix_ptr(matrix);
  // Get dimensions of the matrix
  const int rows = matrix_ptr->get_rows(mesh->order_int);
  const int cols = matrix_ptr->get_cols(mesh->order_int);
  // Check that both op_dats are defined on the same set
  if(x->set->index != y->set->index)
    dg_abort("op_dats passed to op2_gemv are defined on different op_sets");
  const int n = x->set->size;

  #if defined(OP2_DG_CUDA) || defined(OP2_DG_HIP)
  custom_kernel_gemv_sp(x->set, transpose, rows, cols, alpha, beta, matrix_ptr->get_mat_ptr_sp_device(mesh->order_int), x, y);
  #else
  op2_cpu_gemm_sp(transpose ? cols : rows, n, transpose ? rows : cols, alpha, transpose, matrix_ptr->get_mat_ptr_sp_device(mesh->order_int), rows, x, x->dim, beta, y, y->dim);
  #endif
}

void op2_gemv_halo_exchange_sp(DGMesh *mesh, bool transpose, const DG_FP alpha,
              DGConstants::Constant_Matrix matrix, op_dat x, const DG_FP beta,
              op_dat y) {
  // Get the DGConstantMatrix
  DGConstantMatrix *matrix_ptr = constants->get_dg_constant_matrix_ptr(matrix);
  // Get dimensions of the matrix
  const int rows = matrix_ptr->get_rows(mesh->order_int);
  const int cols = matrix_ptr->get_cols(mesh->order_int);
  // Check that both op_dats are defined on the same set
  if(x->set->index != y->set->index)
    dg_abort("op_dats passed to op2_gemv_halo_exchange are defined on different op_sets");

  #if defined(OP2_DG_CUDA) || defined(OP2_DG_HIP)
  custom_kernel_gemv_halo_exchange_sp(x->set, transpose, rows, cols, alpha, beta, matrix_ptr->get_mat_ptr_sp_device(mesh->order_int), x, y);
  #else
  op2_cpu_gemm_halo_exchange_sp(transpose ? cols : rows, transpose ? rows : cols, alpha, transpose, matrix_ptr->get_mat_ptr_sp_device(mesh->order_int), rows, x, x->dim, beta, y, y->dim);
  #endif
}

void op2_gemv_interp_sp(DGMesh *mesh, const int from_N, const int to_N, op_dat x, op_dat y) {
  if(from_N == to_N) {
    op_par_loop(copy_dg_np_sp_tk, "copy_dg_np_sp_tk", mesh->cells,
                op_arg_dat(x, -1, OP_ID, DG_NP, "float", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, "float", OP_WRITE));
    return;
  }

  const int from_NP = DG_CONSTANTS_TK[(from_N - 1) * DG_NUM_CONSTANTS];
  const int to_NP = DG_CONSTANTS_TK[(to_N - 1) * DG_NUM_CONSTANTS];
  const int m = to_NP;
  const int n = mesh->cells->size;
  const int k = from_NP;
  const float *A = constants->get_mat_ptr_device_sp(DGConstants::INTERP_MATRIX_ARRAY) + ((from_N - 1) * DG_ORDER + (to_N - 1)) * DG_NP * DG_NP;

  #if defined(OP2_DG_CUDA) || defined(OP2_DG_HIP)
  custom_kernel_gemv_sp(mesh->cells, false, m, k, 1.0, 0.0, A, x, y);
  #else
  op2_cpu_gemm_sp(m, n, k, 1.0, false, A, m, x, DG_NP, 0.0, y, DG_NP);
  #endif
}
