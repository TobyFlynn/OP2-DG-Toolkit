#include "dg_op2_blas.h"

#include "op_seq.h"

#include "dg_compiler_defs.h"

#include <iostream>
#include <stdexcept>

extern DGConstants *constants;

#if DG_DIM == 2
#include "dg_global_constants/dg_global_constants_2d.h"
#else
#include "dg_global_constants/dg_global_constants_3d.h"
#include "dg_mesh/dg_mesh_3d.h"
#endif

#ifndef OP2_DG_CUDA
#ifdef OP2_DG_USE_LIBXSMM
#include "libxsmm_source.h"
#else
#include "cblas.h"
#endif

// Assumes all elements are same order (does not work for p-adaptivity)
void op2_cpu_gemm(const int m, const int n, const int k,
                  const DG_FP alpha, const bool trans, const DG_FP *A,
                  const int lda, op_dat b_dat, const int ldb, const DG_FP beta,
                  op_dat c_dat, const int ldc) {
  op_arg args[] = {
    op_arg_dat(b_dat, -1, OP_ID, b_dat->dim, DG_FP_STR, OP_READ),
    op_arg_dat(c_dat, -1, OP_ID, c_dat->dim, DG_FP_STR, OP_RW)
  };
  op_mpi_halo_exchanges(b_dat->set, 2, args);
  op_mpi_wait_all(2, args);

  const DG_FP *B = (DG_FP *)b_dat->data;
  DG_FP *C = (DG_FP *)c_dat->data;
  #ifdef OP2_DG_USE_LIBXSMM
  const int flags = trans ? LIBXSMM_GEMM_FLAG_TRANS_A : LIBXSMM_GEMM_FLAG_NONE;
  const int prefetch = LIBXSMM_PREFETCH_NONE;
  const int batch_size = 32;
  const char transA = trans ? 't' : 'n';
  const char noTrans = 'n';

  libxsmm_dmmfunction xmm = NULL;
  if(!trans && alpha == 1.0 && (beta == 1.0 || beta == 0.0))
    xmm = libxsmm_dmmdispatch(m, batch_size, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch);

  if(xmm) {
    #pragma omp parallel for
    for(int i = 0; i < n / batch_size; i++) {
      const DG_FP *_B = B + i * batch_size * ldb;
      DG_FP *_C = C + i * batch_size * ldc;
      // xmm(A, _B, _C, A, _B + batch_size * DG_NUM_FACES * DG_NPF, _C + batch_size * DG_NP);
      xmm(A, _B, _C);
    }
  } else {
    #pragma omp parallel for
    for(int i = 0; i < n / batch_size; i++) {
      const DG_FP *_B = B + i * batch_size * ldb;
      DG_FP *_C = C + i * batch_size * ldc;
      libxsmm_dgemm(&transA, &noTrans, &m, &batch_size, &k, &alpha, A, &lda, _B, &ldb, &beta, _C, &ldc);
    }
  }

  if((n / batch_size) * batch_size != n) {
    const DG_FP *_B = B + ((n / batch_size) * batch_size) * ldb;
    DG_FP *_C = C + ((n / batch_size) * batch_size) * ldc;
    const int final_n = n - ((n / batch_size) * batch_size);
    libxsmm_dgemm(&transA, &noTrans, &m, &final_n, &k, &alpha, A, &lda, _B, &ldb, &beta, _C, &ldc);
  }
  #else
  const int batch_size = n / 32;
  const int num_batches = n / batch_size;
  const CBLAS_TRANSPOSE transA = trans ? CblasTrans : CblasNoTrans;
  #pragma omp parallel for
  for(int i = 0; i < num_batches; i++) {
    cblas_dgemm(CblasColMajor, transA, CblasNoTrans, m, batch_size, k, alpha, A, lda, B + i * batch_size * ldb, ldb, beta, C + i * batch_size * ldc, ldc);
  }
  if(n != batch_size * num_batches) {
    const int left_over_n = n - batch_size * num_batches;
    cblas_dgemm(CblasColMajor, transA, CblasNoTrans, m, left_over_n, k, alpha, A, lda, B + num_batches * batch_size * ldb, ldb, beta, C + num_batches * batch_size * ldc, ldc);
  }
  #endif
  op_mpi_set_dirtybit(2, args);
}

void op2_cpu_gemm_halo_exchange(const int m, const int k,
                  const DG_FP alpha, const bool trans, const DG_FP *A,
                  const int lda, op_dat b_dat, const int ldb, const DG_FP beta,
                  op_dat c_dat, const int ldc) {
  op_arg args[] = {
    op_arg_dat(b_dat, -1, OP_ID, b_dat->dim, DG_FP_STR, OP_READ),
    op_arg_dat(c_dat, -1, OP_ID, c_dat->dim, DG_FP_STR, beta == 0.0 ? OP_WRITE : OP_RW)
  };
  op_mpi_halo_exchanges_grouped(b_dat->set, 2, args, 1, 1);

  for(int round = 0; round < 2; round++) {
    if(round == 1)
      op_mpi_wait_all_grouped(2, args, 1, 1);
    const int n = round == 0 ? b_dat->set->size : b_dat->set->size + b_dat->set->exec_size + b_dat->set->nonexec_size;
    const int start = round == 0 ? 0 : b_dat->set->size;
    const int round_size = n - start;
    const DG_FP *B = (DG_FP *)b_dat->data + start * ldb;
    DG_FP *C = (DG_FP *)c_dat->data + start * ldc;
    #ifdef OP2_DG_USE_LIBXSMM
    const int flags = trans ? LIBXSMM_GEMM_FLAG_TRANS_A : LIBXSMM_GEMM_FLAG_NONE;
    const int prefetch = LIBXSMM_PREFETCH_NONE;
    const int batch_size = 32;
    const char transA = trans ? 't' : 'n';
    const char noTrans = 'n';

    libxsmm_dmmfunction xmm = NULL;
    if(!trans && alpha == 1.0 && (beta == 1.0 || beta == 0.0))
      xmm = libxsmm_dmmdispatch(m, batch_size, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch);

    if(xmm) {
      #pragma omp parallel for
      for(int i = 0; i < round_size / batch_size; i++) {
        const DG_FP *_B = B + i * batch_size * ldb;
        DG_FP *_C = C + i * batch_size * ldc;
        // xmm(A, _B, _C, A, _B + batch_size * DG_NUM_FACES * DG_NPF, _C + batch_size * DG_NP);
        xmm(A, _B, _C);
      }
    } else {
      #pragma omp parallel for
      for(int i = 0; i < round_size / batch_size; i++) {
        const DG_FP *_B = B + i * batch_size * ldb;
        DG_FP *_C = C + i * batch_size * ldc;
        libxsmm_dgemm(&transA, &noTrans, &m, &batch_size, &k, &alpha, A, &lda, _B, &ldb, &beta, _C, &ldc);
      }
    }

    if((round_size / batch_size) * batch_size != round_size) {
      const DG_FP *_B = B + ((round_size / batch_size) * batch_size) * ldb;
      DG_FP *_C = C + ((round_size / batch_size) * batch_size) * ldc;
      const int final_n = round_size - ((round_size / batch_size) * batch_size);
      libxsmm_dgemm(&transA, &noTrans, &m, &final_n, &k, &alpha, A, &lda, _B, &ldb, &beta, _C, &ldc);
    }
    #else
    const int batch_size = round_size / 32;
    const int num_batches = round_size / batch_size;
    const CBLAS_TRANSPOSE transA = trans ? CblasTrans : CblasNoTrans;
    #pragma omp parallel for
    for(int i = 0; i < num_batches; i++) {
      cblas_dgemm(CblasColMajor, transA, CblasNoTrans, m, batch_size, k, alpha, A, lda, B + i * batch_size * ldb, ldb, beta, C + i * batch_size * ldc, ldc);
    }
    if(round_size != batch_size * num_batches) {
      const int left_over_n = round_size - batch_size * num_batches;
      cblas_dgemm(CblasColMajor, transA, CblasNoTrans, m, left_over_n, k, alpha, A, lda, B + num_batches * batch_size * ldb, ldb, beta, C + num_batches * batch_size * ldc, ldc);
    }
    #endif
  }
  op_mpi_set_dirtybit_force_halo_exchange(2, args, 1);
}

void op2_cpu_gemm_sp(const int m, const int n, const int k,
                  const float alpha, const bool trans, const float *A_sp,
                  const int lda, op_dat b_dat, const int ldb, const float beta,
                  op_dat c_dat, const int ldc) {
  op_arg args[] = {
    op_arg_dat(b_dat, -1, OP_ID, b_dat->dim, "float", OP_READ),
    op_arg_dat(c_dat, -1, OP_ID, c_dat->dim, "float", OP_RW)
  };
  op_mpi_halo_exchanges(b_dat->set, 2, args);
  op_mpi_wait_all(2, args);

  const float *B = (float *)b_dat->data;
  float *C = (float *)c_dat->data;

  const int batch_size = n / 32;
  const int num_batches = n / batch_size;
  const CBLAS_TRANSPOSE transA = trans ? CblasTrans : CblasNoTrans;
  #pragma omp parallel for
  for(int i = 0; i < num_batches; i++) {
    cblas_sgemm(CblasColMajor, transA, CblasNoTrans, m, batch_size, k, alpha, A_sp, lda, B + i * batch_size * ldb, ldb, beta, C + i * batch_size * ldc, ldc);
  }
  if(n != batch_size * num_batches) {
    const int left_over_n = n - batch_size * num_batches;
    cblas_sgemm(CblasColMajor, transA, CblasNoTrans, m, left_over_n, k, alpha, A_sp, lda, B + num_batches * batch_size * ldb, ldb, beta, C + num_batches * batch_size * ldc, ldc);
  }

  op_mpi_set_dirtybit(2, args);
}

void op2_cpu_gemm_halo_exchange_sp(const int m, const int k,
                  const float alpha, const bool trans, const float *A_sp,
                  const int lda, op_dat b_dat, const int ldb, const float beta,
                  op_dat c_dat, const int ldc) {
  op_arg args[] = {
    op_arg_dat(b_dat, -1, OP_ID, b_dat->dim, "float", OP_READ),
    op_arg_dat(c_dat, -1, OP_ID, c_dat->dim, "float", beta == 0.0 ? OP_WRITE : OP_RW)
  };
  op_mpi_halo_exchanges_grouped(b_dat->set, 2, args, 1, 1);

  const float *B = (float *)b_dat->data;
  float *C = (float *)c_dat->data;

  for(int round = 0; round < 2; round++) {
    if(round == 1)
      op_mpi_wait_all_grouped(2, args, 1, 1);

    const int n = round == 0 ? b_dat->set->size : b_dat->set->size + b_dat->set->exec_size + b_dat->set->nonexec_size;
    const int start = round == 0 ? 0 : b_dat->set->size;
    const int round_size = n - start;

    const float *B = (float *)b_dat->data + start * ldb;
    float *C = (float *)c_dat->data + start * ldc;

    const int batch_size = round_size / 32;
    const int num_batches = round_size / batch_size;
    const CBLAS_TRANSPOSE transA = trans ? CblasTrans : CblasNoTrans;
    #pragma omp parallel for
    for(int i = 0; i < num_batches; i++) {
      cblas_sgemm(CblasColMajor, transA, CblasNoTrans, m, batch_size, k, alpha, A_sp, lda, B + i * batch_size * ldb, ldb, beta, C + i * batch_size * ldc, ldc);
    }
    if(round_size != batch_size * num_batches) {
      const int left_over_n = round_size - batch_size * num_batches;
      cblas_sgemm(CblasColMajor, transA, CblasNoTrans, m, left_over_n, k, alpha, A_sp, lda, B + num_batches * batch_size * ldb, ldb, beta, C + num_batches * batch_size * ldc, ldc);
    }
  }

  op_mpi_set_dirtybit_force_halo_exchange(2, args, 1);
}

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

void op2_gemv_inv_mass_gass_interpT(DGMesh *mesh, bool transpose,
                                    const DG_FP alpha, op_dat x,
                                    const DG_FP beta, op_dat y) {
  if(transpose) {
    std::cerr << "op2_gemv_inv_mass_gass_interpT not implemented for transpose ... exiting" << std::endl;
  } else {
    op_par_loop(gemv_inv_mass_gauss_interpT, "gemv_inv_mass_gauss_interpT", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, DG_FP_STR, OP_READ),
                op_arg_gbl(&beta,  1, DG_FP_STR, OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::INV_MASS_GAUSS_INTERP_T), DG_ORDER * DG_G_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->J, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_G_NP, DG_FP_STR, OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  }
}

void op2_gemv_gauss_interp(DGMesh *mesh, bool transpose, const DG_FP alpha,
                           op_dat x, const DG_FP beta, op_dat y) {
  if(transpose) {
    op_par_loop(gemv_gauss_interpT, "gemv_gauss_interpT", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, DG_FP_STR, OP_READ),
                op_arg_gbl(&beta,  1, DG_FP_STR, OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_INTERP), DG_ORDER * DG_G_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_G_NP, DG_FP_STR, OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  } else {
    op_par_loop(gemv_gauss_interp, "gemv_gauss_interp", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, DG_FP_STR, OP_READ),
                op_arg_gbl(&beta,  1, DG_FP_STR, OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_INTERP), DG_ORDER * DG_G_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_G_NP, DG_FP_STR, OP_RW));
  }
}

void op2_gemv_lift(DGMesh *mesh, bool transpose, const DG_FP alpha, op_dat x,
                   const DG_FP beta, op_dat y) {
  #if defined(USE_OP2_KERNELS)
  if(transpose) {
    op_par_loop(gemv_liftT, "gemv_liftT", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, DG_FP_STR, OP_READ),
                op_arg_gbl(&beta,  1, DG_FP_STR, OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::LIFT), DG_ORDER * DG_NUM_FACES * DG_NPF * DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_RW));
  } else {
    op_par_loop(gemv_lift, "gemv_lift", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, DG_FP_STR, OP_READ),
                op_arg_gbl(&beta,  1, DG_FP_STR, OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::LIFT), DG_ORDER * DG_NUM_FACES * DG_NPF * DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  }
  #elif defined(OP2_DG_CUDA)
  const int order = mesh->order_int;
  const int m = DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS];
  const int k = DG_NUM_FACES * DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS + 1];
  const DG_FP *A = constants->get_mat_ptr_kernel(DGConstants::LIFT) + (order - 1) * DG_NUM_FACES * DG_NPF * DG_NP;
  custom_kernel_gemv(mesh->cells, transpose, m, k, alpha, beta, A, x, y);
  #else
  const int order = mesh->order_int;
  const int m = DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS];
  const int n = mesh->cells->size;
  const int k = DG_NUM_FACES * DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS + 1];
  const DG_FP *A = constants->get_mat_ptr_kernel(DGConstants::LIFT) + (order - 1) * DG_NUM_FACES * DG_NPF * DG_NP;
  op2_cpu_gemm(m, n, k, alpha, transpose, A, m, x, DG_NUM_FACES * DG_NPF, beta, y, DG_NP);
  #endif
}

void op2_gemv_emat(DGMesh *mesh, bool transpose, const DG_FP alpha, op_dat x,
                   const DG_FP beta, op_dat y) {
  #if defined(USE_OP2_KERNELS)
  if(transpose) {
    op_par_loop(gemv_liftT, "gemv_liftT", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, DG_FP_STR, OP_READ),
                op_arg_gbl(&beta,  1, DG_FP_STR, OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::EMAT), DG_ORDER * DG_NUM_FACES * DG_NPF * DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_RW));
  } else {
    op_par_loop(gemv_lift, "gemv_lift", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, DG_FP_STR, OP_READ),
                op_arg_gbl(&beta,  1, DG_FP_STR, OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::EMAT), DG_ORDER * DG_NUM_FACES * DG_NPF * DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  }
  #elif defined(OP2_DG_CUDA)
  const int order = mesh->order_int;
  const int m = DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS];
  const int k = DG_NUM_FACES * DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS + 1];
  const DG_FP *A = constants->get_mat_ptr_kernel(DGConstants::EMAT) + (order - 1) * DG_NUM_FACES * DG_NPF * DG_NP;
  custom_kernel_gemv(mesh->cells, transpose, m, k, alpha, beta, A, x, y);
  #else
  const int order = mesh->order_int;
  const int m = DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS];
  const int n = mesh->cells->size;
  const int k = DG_NUM_FACES * DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS + 1];
  const DG_FP *A = constants->get_mat_ptr_kernel(DGConstants::EMAT) + (order - 1) * DG_NUM_FACES * DG_NPF * DG_NP;
  op2_cpu_gemm(m, n, k, alpha, transpose, A, m, x, DG_NUM_FACES * DG_NPF, beta, y, DG_NP);
  #endif
}

void op2_gemv_np_np(DGMesh *mesh, bool transpose, const DG_FP alpha,
                    const DG_FP *matrix, op_dat x, const DG_FP beta,
                    op_dat y) {
  #if defined(USE_OP2_KERNELS)
  if(transpose) {
    op_par_loop(gemv_np_npT, "gemv_np_npT", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, DG_FP_STR, OP_READ),
                op_arg_gbl(&beta,  1, DG_FP_STR, OP_READ),
                op_arg_gbl(matrix, DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  } else {
    op_par_loop(gemv_np_np, "gemv_np_np", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, DG_FP_STR, OP_READ),
                op_arg_gbl(&beta,  1, DG_FP_STR, OP_READ),
                op_arg_gbl(matrix, DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  }
  #elif defined(OP2_DG_CUDA)
  const int order = mesh->order_int;
  const int m = DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS];
  const int k = m;
  const DG_FP *A = matrix + (order - 1) * DG_NP * DG_NP;
  custom_kernel_gemv(mesh->cells, transpose, m, k, alpha, beta, A, x, y);
  #else
  const int order = mesh->order_int;
  const int m = DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS];
  const int n = mesh->cells->size;
  const int k = m;
  const DG_FP *A = matrix + (order - 1) * DG_NP * DG_NP;
  op2_cpu_gemm(m, n, k, alpha, transpose, A, m, x, DG_NP, beta, y, DG_NP);
  #endif
}

void op2_gemv_cub_np_np(DGMesh *mesh, bool transpose, const DG_FP alpha,
                        const DG_FP *matrix, op_dat x, const DG_FP beta,
                        op_dat y) {
  if(transpose) {
    op_par_loop(gemv_cub_np_npT, "gemv_cub_np_npT", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, DG_FP_STR, OP_READ),
                op_arg_gbl(&beta,  1, DG_FP_STR, OP_READ),
                op_arg_gbl(matrix, DG_ORDER * DG_CUB_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_CUB_NP, DG_FP_STR, OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  } else {
    op_par_loop(gemv_cub_np_np, "gemv_cub_np_np", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, DG_FP_STR, OP_READ),
                op_arg_gbl(&beta,  1, DG_FP_STR, OP_READ),
                op_arg_gbl(matrix, DG_ORDER * DG_CUB_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_CUB_NP, DG_FP_STR, OP_RW));
  }
}

void op2_gemv_cub3d_np_np(DGMesh *mesh, bool transpose, const DG_FP alpha, const DG_FP *matrix, 
                          op_dat x, const DG_FP beta, op_dat y) {
  const int m = DG_CUB_3D_NP;
  const int k = DG_NP;
  #if defined(USE_OP2_KERNELS)
  throw std::runtime_error("op2_gemv_cub3d_np_np does not have a OP2 kernel implementation");
  #elif defined(OP2_DG_CUDA)
  custom_kernel_gemv(mesh->cells, transpose, m, k, alpha, beta, matrix, x, y);
  #else
  const int n = mesh->cells->size;
  op2_cpu_gemm(m, n, k, alpha, transpose, matrix, m, x, x->dim, beta, y, y->dim);
  #endif
}

void op2_gemv_np_cub3d_np(DGMesh *mesh, bool transpose, const DG_FP alpha, const DG_FP *matrix, 
                          op_dat x, const DG_FP beta, op_dat y) {
  const int m = DG_NP;
  const int k = DG_CUB_3D_NP;
  #if defined(USE_OP2_KERNELS)
  throw std::runtime_error("op2_gemv_cub3d_np_np does not have a OP2 kernel implementation");
  #elif defined(OP2_DG_CUDA)
  custom_kernel_gemv(mesh->cells, transpose, m, k, alpha, beta, matrix, x, y);
  #else
  const int n = mesh->cells->size;
  op2_cpu_gemm(m, n, k, alpha, transpose, matrix, m, x, x->dim, beta, y, y->dim);
  #endif
}

void op2_gemv(DGMesh *mesh, bool transpose, const DG_FP alpha,
              DGConstants::Constant_Matrix matrix, op_dat x, const DG_FP beta,
              op_dat y) {
  switch(matrix) {
    case DGConstants::DR:
    case DGConstants::DS:
    case DGConstants::DT:
    case DGConstants::DRW:
    case DGConstants::DSW:
    case DGConstants::DTW:
    case DGConstants::MASS:
    case DGConstants::INV_MASS:
    case DGConstants::V:
    case DGConstants::INV_V:
      op2_gemv_np_np(mesh, transpose, alpha, constants->get_mat_ptr_kernel(matrix), x, beta, y);
      break;
    case DGConstants::CUB_V:
      #if DG_DIM == 2
      op2_gemv_cub_np_np(mesh, transpose, alpha, constants->get_mat_ptr(matrix), x, beta, y);
      #else
      op2_gemv_cub_np_np(mesh, transpose, alpha, constants->get_mat_ptr_kernel(matrix), x, beta, y);
      #endif
      break;
    case DGConstants::LIFT:
      op2_gemv_lift(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::EMAT:
      op2_gemv_emat(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::INV_MASS_GAUSS_INTERP_T:
      op2_gemv_inv_mass_gass_interpT(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::GAUSS_INTERP:
      op2_gemv_gauss_interp(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::CUB_VDR:
    case DGConstants::CUB_VDS:
    case DGConstants::CUB_DR:
    case DGConstants::CUB_DS:
      op2_gemv_cub_np_np(mesh, transpose, alpha, constants->get_mat_ptr(matrix), x, beta, y);
      break;
    case DGConstants::CUB3D_INTERP:
      op2_gemv_cub3d_np_np(mesh, transpose, alpha, constants->get_mat_ptr_kernel(matrix), x, beta, y);
      break;
    case DGConstants::CUB3D_PROJ:
    case DGConstants::CUB3D_PDR:
    case DGConstants::CUB3D_PDS:
    case DGConstants::CUB3D_PDT:
      op2_gemv_np_cub3d_np(mesh, transpose, alpha, constants->get_mat_ptr_kernel(matrix), x, beta, y);
      break;
    default:
      std::cerr << "op2_gemv call not implemented for this matrix ... exiting" << std::endl;
      exit(2);
  }
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
  const DG_FP *A = constants->get_mat_ptr_kernel(DGConstants::INTERP_MATRIX_ARRAY) + ((from_N - 1) * DG_ORDER + (to_N - 1)) * DG_NP * DG_NP;

  // TODO 2D
  #if defined(USE_OP2_KERNELS)
  op_par_loop(interp_dat_to_new_order_3d_copy, "interp_dat_to_new_order_3d_copy", mesh->cells,
              op_arg_gbl(constants->get_mat_ptr_kernel(DGConstants::INTERP_MATRIX_ARRAY), DG_ORDER * DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(&from_N, 1, "int", OP_READ),
              op_arg_gbl(&to_N, 1, "int", OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));
  #elif defined(OP2_DG_CUDA)
  custom_kernel_gemv(mesh->cells, false, m, k, 1.0, 0.0, A, x, y);
  #else
  op2_cpu_gemm(m, n, k, 1.0, false, A, m, x, DG_NP, 0.0, y, DG_NP);
  #endif
}

void op2_gemv_np_np_halo_exchange(DGMesh *mesh, bool transpose, const DG_FP alpha,
                    const DG_FP *matrix, op_dat x, const DG_FP beta,
                    op_dat y) {
  #if defined(USE_OP2_KERNELS) || (defined(OP2_DG_CUDA) && DG_DIM == 2)
  throw std::runtime_error("gemv_halo_exchange not fully supported yet\n");
  #elif defined(OP2_DG_CUDA)
  const int order = ((DGMesh3D *)mesh)->order_int;
  const int m = DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS];
  const int k = m;
  const DG_FP *A = matrix + (order - 1) * DG_NP * DG_NP;
  custom_kernel_gemv_halo_exchange(mesh->cells, transpose, m, k, alpha, beta, A, x, y);
  #else
  const int order = *((int *)mesh->order->data);
  const int m = DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS];
  const int k = m;
  const DG_FP *A = matrix + (order - 1) * DG_NP * DG_NP;
  op2_cpu_gemm_halo_exchange(m, k, alpha, transpose, A, m, x, DG_NP, beta, y, DG_NP);
  #endif
}

void op2_gemv_halo_exchange(DGMesh *mesh, bool transpose, const DG_FP alpha,
              DGConstants::Constant_Matrix matrix, op_dat x, const DG_FP beta,
              op_dat y) {
  switch(matrix) {
    case DGConstants::DR:
    case DGConstants::DS:
    case DGConstants::DT:
      op2_gemv_np_np_halo_exchange(mesh, transpose, alpha, constants->get_mat_ptr_kernel(matrix), x, beta, y);
      break;
    default:
      std::cerr << "op2_gemv call not implemented for this matrix ... exiting" << std::endl;
      exit(2);
  }
}

void op2_gemv_lift_sp(DGMesh *mesh, bool transpose, const DG_FP alpha, op_dat x,
                   const DG_FP beta, op_dat y) {
  #if defined(USE_OP2_KERNELS)
  throw std::runtime_error("SP");
  #elif defined(OP2_DG_CUDA)
  const int order = mesh->order_int;
  const int m = DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS];
  const int k = DG_NUM_FACES * DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS + 1];
  const float *A = constants->get_mat_ptr_kernel_sp(DGConstants::LIFT) + (order - 1) * DG_NUM_FACES * DG_NPF * DG_NP;
  custom_kernel_gemv_sp(mesh->cells, transpose, m, k, alpha, beta, A, x, y);
  #else
  const int order = mesh->order_int;
  const int m = DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS];
  const int n = mesh->cells->size;
  const int k = DG_NUM_FACES * DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS + 1];
  const float *A = constants->get_mat_ptr_kernel_sp(DGConstants::LIFT) + (order - 1) * DG_NUM_FACES * DG_NPF * DG_NP;
  op2_cpu_gemm_sp(m, n, k, alpha, transpose, A, m, x, DG_NUM_FACES * DG_NPF, beta, y, DG_NP);
  #endif
}

void op2_gemv_emat_sp(DGMesh *mesh, bool transpose, const DG_FP alpha, op_dat x,
                   const DG_FP beta, op_dat y) {
  #if defined(USE_OP2_KERNELS)
  throw std::runtime_error("SP");
  #elif defined(OP2_DG_CUDA)
  const int order = mesh->order_int;
  const int m = DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS];
  const int k = DG_NUM_FACES * DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS + 1];
  const float *A = constants->get_mat_ptr_kernel_sp(DGConstants::EMAT) + (order - 1) * DG_NUM_FACES * DG_NPF * DG_NP;
  custom_kernel_gemv_sp(mesh->cells, transpose, m, k, alpha, beta, A, x, y);
  #else
  const int order = mesh->order_int;
  const int m = DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS];
  const int n = mesh->cells->size;
  const int k = DG_NUM_FACES * DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS + 1];
  const float *A = constants->get_mat_ptr_kernel_sp(DGConstants::EMAT) + (order - 1) * DG_NUM_FACES * DG_NPF * DG_NP;
  op2_cpu_gemm_sp(m, n, k, alpha, transpose, A, m, x, DG_NUM_FACES * DG_NPF, beta, y, DG_NP);
  #endif
}

void op2_gemv_np_np_sp(DGMesh *mesh, bool transpose, const DG_FP alpha,
                    const float *matrix, op_dat x, const DG_FP beta,
                    op_dat y) {
  #if defined(USE_OP2_KERNELS)
  throw std::runtime_error("SP");
  #elif defined(OP2_DG_CUDA)
  const int order = mesh->order_int;
  const int m = DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS];
  const int k = m;
  const float *A = matrix + (order - 1) * DG_NP * DG_NP;
  custom_kernel_gemv_sp(mesh->cells, transpose, m, k, alpha, beta, A, x, y);
  #else
  const int order = mesh->order_int;
  const int m = DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS];
  const int n = mesh->cells->size;
  const int k = m;
  const float *A = matrix + (order - 1) * DG_NP * DG_NP;
  op2_cpu_gemm_sp(m, n, k, alpha, transpose, A, m, x, DG_NP, beta, y, DG_NP);
  #endif
}

void op2_gemv_sp(DGMesh *mesh, bool transpose, const DG_FP alpha,
              DGConstants::Constant_Matrix matrix, op_dat x, const DG_FP beta,
              op_dat y) {
  switch(matrix) {
    case DGConstants::DR:
    case DGConstants::DS:
    case DGConstants::DT:
    case DGConstants::DRW:
    case DGConstants::DSW:
    case DGConstants::DTW:
    case DGConstants::MASS:
    case DGConstants::INV_MASS:
    case DGConstants::V:
    case DGConstants::INV_V:
      op2_gemv_np_np_sp(mesh, transpose, alpha, constants->get_mat_ptr_kernel_sp(matrix), x, beta, y);
      break;
    case DGConstants::LIFT:
      op2_gemv_lift_sp(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::EMAT:
      op2_gemv_emat_sp(mesh, transpose, alpha, x, beta, y);
      break;
    default:
      std::cerr << "op2_gemv_sp call not implemented for this matrix ... exiting" << std::endl;
      exit(2);
  }
}

void op2_gemv_np_np_halo_exchange_sp(DGMesh *mesh, bool transpose, const DG_FP alpha,
                    const float *matrix, op_dat x, const DG_FP beta,
                    op_dat y) {
  #if defined(USE_OP2_KERNELS) || (defined(OP2_DG_CUDA) && DG_DIM == 2)
  throw std::runtime_error("gemv_halo_exchange not fully supported yet\n");
  #elif defined(OP2_DG_CUDA)
  const int order = mesh->order_int;
  const int m = DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS];
  const int k = m;
  const float *A = matrix + (order - 1) * DG_NP * DG_NP;
  custom_kernel_gemv_halo_exchange_sp(mesh->cells, transpose, m, k, alpha, beta, A, x, y);
  #else
  const int order = mesh->order_int;
  const int m = DG_CONSTANTS_TK[(order - 1) * DG_NUM_CONSTANTS];
  const int k = m;
  const float *A = matrix + (order - 1) * DG_NP * DG_NP;
  op2_cpu_gemm_halo_exchange_sp(m, k, alpha, transpose, A, m, x, DG_NP, beta, y, DG_NP);
  #endif
}

void op2_gemv_halo_exchange_sp(DGMesh *mesh, bool transpose, const DG_FP alpha,
              DGConstants::Constant_Matrix matrix, op_dat x, const DG_FP beta,
              op_dat y) {
  switch(matrix) {
    case DGConstants::DR:
    case DGConstants::DS:
    case DGConstants::DT:
      op2_gemv_np_np_halo_exchange_sp(mesh, transpose, alpha, constants->get_mat_ptr_kernel_sp(matrix), x, beta, y);
      break;
    default:
      std::cerr << "op2_gemv call not implemented for this matrix ... exiting" << std::endl;
      exit(2);
  }
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
  const float *A = constants->get_mat_ptr_kernel_sp(DGConstants::INTERP_MATRIX_ARRAY) + ((from_N - 1) * DG_ORDER + (to_N - 1)) * DG_NP * DG_NP;

  #if defined(OP2_DG_CUDA)
  custom_kernel_gemv_sp(mesh->cells, false, m, k, 1.0, 0.0, A, x, y);
  #else
  op2_cpu_gemm_sp(m, n, k, 1.0, false, A, m, x, DG_NP, 0.0, y, DG_NP);
  #endif
}
