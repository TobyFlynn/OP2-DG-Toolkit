#ifndef __DG_BLAS_CALLS_H
#define __DG_BLAS_CALLS_H

#include "op_seq.h"
#include "dg_mesh.h"
#include "dg_constants.h"

extern DGConstants *constants;

void init_grid_blas(DGMesh *mesh);

void init_gauss_blas(DGMesh *mesh, DGGaussData *gaussData);

// Assumes matrix is in column major form and both op_dat are defined on the same set
void op2_gemv(bool transpose, int m, int n, double alpha, double *A_ptr, int lda, op_dat x, double beta, op_dat y);

void op2_gemm(bool transposeA, bool transposeB, int m, int n, int k, double alpha, double *a_ptr, int lda, op_dat b, int ldb, double beta, op_dat c, int ldc);

void op2_gemm(bool transposeA, bool transposeB, int m, int n, int k, double alpha, op_dat a, int lda, double *b_ptr, int ldb, double beta, op_dat c, int ldc);

void op2_gemv_batch(bool transpose, int m, int n, double alpha, op_dat a, int lda, op_dat x, double beta, op_dat y);

void op2_gemm_batch(bool transposeA, bool transposeB, int m, int n, int k, double alpha, op_dat a, int lda, op_dat b, int ldb, double beta, op_dat c, int ldc);

#endif
