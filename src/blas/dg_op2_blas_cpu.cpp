#include "op_seq.h"

#include "dg_compiler_defs.h"
#include "dg_abort.h"

#ifdef OP2_DG_USE_LIBXSMM
#include "libxsmm_source.h"
#else
#include "cblas.h"
#endif

#if DG_DIM == 2
#include "dg_global_constants/dg_global_constants_2d.h"
#else
#include "dg_global_constants/dg_global_constants_3d.h"
#include "dg_mesh/dg_mesh_3d.h"
#endif

void init_op2_gemv() {
  #ifdef OP2_DG_USE_LIBXSMM
  libxsmm_init();
  #endif
}

void destroy_op2_gemv() {
  #ifdef OP2_DG_USE_LIBXSMM
  libxsmm_finalize();
  #endif
}

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
    if(round_size <= 0)
      continue;
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
    if(round_size <= 0)
      continue;

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