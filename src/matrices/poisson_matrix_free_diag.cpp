#include "dg_matrices/poisson_matrix_free_diag.h"

#include "op_seq.h"

#include "dg_abort.h"
#include "timing.h"

extern Timing *timer;

bool PoissonMatrixFreeDiag::getPETScMat(Mat** mat) {
  dg_abort("Not able to get PETSc Matrices for Semi Matrix Free class\n");
  return false;
}

void PoissonMatrixFreeDiag::calc_mat() {
  dg_abort("calc_mat has not been implemented for this Semi Matrix Free class\n");
}

void PoissonMatrixFreeDiag::mult(op_dat in, op_dat out) {
  dg_abort("mult has not been implemented for this Semi Matrix Free class\n");
}

void PoissonMatrixFreeDiag::multJacobi(op_dat in, op_dat out) {
  timer->startTimer("PoissonMatrixFreeDiag - multJacobi");
  mult(in, out);

  op_par_loop(poisson_diag_mult_jacobi, "poisson_diag_mult_jacobi", _mesh->cells,
              op_arg_gbl(&_mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(diag, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(out, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  timer->endTimer("PoissonMatrixFreeDiag - multJacobi");
}

void PoissonMatrixFreeDiag::setPETScMatrix() {
  dg_abort("Not able to set PETSc Matrices for Semi Matrix Free class\n");
}

void PoissonMatrixFreeDiag::calc_glb_ind() {
  dg_abort("calc_glb_ind has not been implemented for Matrix Free class\n");
}
