#ifndef __DG_OP2_BLAS_H
#define __DG_OP2_BLAS_H

#include "op_seq.h"
#include "dg_mesh/dg_mesh.h"
#include "dg_constants/dg_constants.h"

// Assumes matrix is in column major form and both op_dat are defined on the same set
void op2_gemv(DGMesh *mesh, bool transpose, const double alpha, DGConstants::Constant_Matrix matrix, op_dat x, const double beta, op_dat y);

#endif
