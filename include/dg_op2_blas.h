#ifndef __DG_OP2_BLAS_H
#define __DG_OP2_BLAS_H

#include "dg_compiler_defs.h"

#include "op_seq.h"
#include "dg_mesh/dg_mesh.h"
#include "dg_constants/dg_constants.h"

// Assumes matrix is in column major form and both op_dat are defined on the same set
void op2_gemv(DGMesh *mesh, bool transpose, const DG_FP alpha, DGConstants::Constant_Matrix matrix, op_dat x, const DG_FP beta, op_dat y);

void op2_gemv_halo_exchange(DGMesh *mesh, bool transpose, const DG_FP alpha, DGConstants::Constant_Matrix matrix, op_dat x, const DG_FP beta, op_dat y);

void op2_gemv_interp(DGMesh *mesh, const int from_N, const int to_N, op_dat x, op_dat y);

void op2_gemv_sp(DGMesh *mesh, bool transpose, const DG_FP alpha, DGConstants::Constant_Matrix matrix, op_dat x, const DG_FP beta, op_dat y);

void op2_gemv_halo_exchange_sp(DGMesh *mesh, bool transpose, const DG_FP alpha, DGConstants::Constant_Matrix matrix, op_dat x, const DG_FP beta, op_dat y);

void op2_gemv_interp_sp(DGMesh *mesh, const int from_N, const int to_N, op_dat x, op_dat y);

#endif
