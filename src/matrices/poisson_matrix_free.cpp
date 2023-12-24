#include "dg_matrices/poisson_matrix_free.h"

#include "dg_abort.h"

bool PoissonMatrixFree::getPETScMat(Mat** mat) {
  dg_abort("Not able to get PETSc Matrices for Matrix Free class\n");
  return false;
}

void PoissonMatrixFree::calc_mat() {
  dg_abort("calc_mat has not been implemented for Matrix Free class\n");
}

void PoissonMatrixFree::mult(op_dat in, op_dat out) {
  dg_abort("mult has not been implemented for this Matrix Free class\n");
}

void PoissonMatrixFree::multJacobi(op_dat in, op_dat out) {
  dg_abort("multJacobi has not been implemented for Matrix Free class\n");
}

void PoissonMatrixFree::setPETScMatrix() {
  dg_abort("Not able to set PETSc Matrices for Matrix Free class\n");
}

void PoissonMatrixFree::calc_op1() {
  dg_abort("calc_op1 has not been implemented for Matrix Free class\n");
}

void PoissonMatrixFree::calc_op2() {
  dg_abort("calc_op2 has not been implemented for Matrix Free class\n");
}

void PoissonMatrixFree::calc_opbc() {
  dg_abort("calc_opbc has not been implemented for Matrix Free class\n");
}

void PoissonMatrixFree::calc_glb_ind() {
  dg_abort("calc_glb_ind has not been implemented for Matrix Free class\n");
}
