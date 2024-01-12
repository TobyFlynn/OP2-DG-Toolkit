#include "dg_matrices/poisson_matrix_free_block_diag.h"

#include "op_seq.h"

#include <stdexcept>

#include "timing.h"

extern Timing *timer;

bool PoissonMatrixFreeBlockDiag::getPETScMat(Mat** mat) {
  throw std::runtime_error("Not able to get PETSc Matrices for PoissonMatrixFreeBlockDiag\n");
  return false;
}

void PoissonMatrixFreeBlockDiag::calc_mat() {
  throw std::runtime_error("calc_mat has not been implemented for this PoissonMatrixFreeBlockDiag\n");
}

void PoissonMatrixFreeBlockDiag::mult(op_dat in, op_dat out) {
  throw std::runtime_error("mult has not been implemented for this PoissonMatrixFreeBlockDiag\n");
}

void PoissonMatrixFreeBlockDiag::multJacobi(op_dat in, op_dat out) {
  timer->startTimer("PoissonMatrixFreeBlockDiag - multJacobi");
  mult(in, out);

  op_par_loop(poisson_block_diag_mult_jacobi, "poisson_block_diag_mult_jacobi", _mesh->cells,
              op_arg_gbl(&_mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(block_diag, -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(out, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  timer->endTimer("PoissonMatrixFreeBlockDiag - multJacobi");
}

void PoissonMatrixFreeBlockDiag::setPETScMatrix() {
  throw std::runtime_error("Not able to set PETSc Matrices for PoissonMatrixFreeBlockDiag\n");
}

void PoissonMatrixFreeBlockDiag::calc_glb_ind() {
  throw std::runtime_error("calc_glb_ind has not been implemented for Matrix Free class\n");
}
