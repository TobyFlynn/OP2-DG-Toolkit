#include "dg_matrices/3d/factor_mm_poisson_matrix_free_diag_3d.h"

#include "op_seq.h"

#include "dg_constants/dg_constants.h"

#include "timing.h"

extern DGConstants *constants;
extern Timing *timer;

FactorMMPoissonMatrixFreeDiag3D::FactorMMPoissonMatrixFreeDiag3D(DGMesh3D *m) : FactorPoissonMatrixFreeDiag3D(m) {

}

void FactorMMPoissonMatrixFreeDiag3D::set_mm_factor(op_dat f) {
  mm_factor = f;
}

void FactorMMPoissonMatrixFreeDiag3D::calc_mat_partial() {
  timer->startTimer("FactorMMPoissonMatrixFreeDiag3D - calc_mat_partial");
  check_current_order();
  calc_op1();
  calc_op2();
  calc_opbc();
  calc_mm();
  petscMatResetRequired = true;
  timer->endTimer("FactorMMPoissonMatrixFreeDiag3D - calc_mat_partial");
}

void FactorMMPoissonMatrixFreeDiag3D::mult(op_dat in, op_dat out) {
  timer->startTimer("FactorMMPoissonMatrixFreeDiag3D - Mult");
  FactorPoissonMatrixFreeDiag3D::mult(in, out);

  timer->startTimer("FactorMMPoissonMatrixFreeDiag3D - Mult MM");
  op_par_loop(fpmf_3d_mult_mm, "fpmf_3d_mult_mm", _mesh->cells,
              op_arg_gbl(&_mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
              op_arg_dat(mm_factor, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(in,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(out, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  timer->endTimer("FactorMMPoissonMatrixFreeDiag3D - Mult MM");
  timer->endTimer("FactorMMPoissonMatrixFreeDiag3D - Mult");
  return;
}

void FactorMMPoissonMatrixFreeDiag3D::calc_mm() {
  timer->startTimer("FactorMMPoissonMatrixFreeDiag3D - calc_mm");
  op_par_loop(factor_poisson_matrix_3d_mm_diag, "factor_poisson_matrix_3d_mm_diag", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mm_factor, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
              op_arg_dat(diag, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  timer->endTimer("FactorMMPoissonMatrixFreeDiag3D - calc_mm");
}
