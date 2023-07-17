#include "dg_matrices/3d/factor_poisson_matrix_free_diag_3d.h"

#include "op_seq.h"

#include "dg_constants/dg_constants.h"
#include "dg_op2_blas.h"

#include "timing.h"

extern DGConstants *constants;
extern Timing *timer;

FactorPoissonMatrixFreeDiag3D::FactorPoissonMatrixFreeDiag3D(DGMesh3D *m) : FactorPoissonMatrixFreeMult3D(m) {
  _mesh = m;

  diag = op_decl_dat(mesh->cells, DG_NP, DG_FP_STR, (DG_FP *)NULL, "poisson_matrix_free_diag");
}

void FactorPoissonMatrixFreeDiag3D::set_bc_types(op_dat bc_ty) {
  bc_types = bc_ty;
  mat_free_set_bc_types(bc_ty);
}

void FactorPoissonMatrixFreeDiag3D::set_factor(op_dat f) {
  factor = f;
  mat_free_set_factor(f);
}

void FactorPoissonMatrixFreeDiag3D::apply_bc(op_dat rhs, op_dat bc) {
  mat_free_apply_bc(rhs, bc);
}

void FactorPoissonMatrixFreeDiag3D::mult(op_dat in, op_dat out) {
  timer->startTimer("FactorPoissonMatrixFreeDiag3D - mult");
  mat_free_mult(in, out);
  timer->endTimer("FactorPoissonMatrixFreeDiag3D - mult");
}

void FactorPoissonMatrixFreeDiag3D::calc_mat_partial() {
  timer->startTimer("FactorPoissonMatrixFreeDiag3D - calc_mat_partial");
  check_current_order();
  calc_op1();
  calc_op2();
  calc_opbc();
  petscMatResetRequired = true;
  timer->endTimer("FactorPoissonMatrixFreeDiag3D - calc_mat_partial");
}

void FactorPoissonMatrixFreeDiag3D::calc_op1() {
  timer->startTimer("FactorPoissonMatrixFreeDiag3D - calc_op1");
  op_par_loop(factor_poisson_matrix_3d_op1_diag, "factor_poisson_matrix_3d_op1_diag", mesh->cells,
              op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(mesh->rx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->sx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->tx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->ry, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->sy, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->ty, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->rz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->sz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->tz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->J,  -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mat_free_factor_copy, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(diag, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));
  timer->endTimer("FactorPoissonMatrixFreeDiag3D - calc_op1");
}

void FactorPoissonMatrixFreeDiag3D::calc_op2() {
  timer->startTimer("FactorPoissonMatrixFreeDiag3D - calc_op2");
  op_par_loop(factor_poisson_matrix_3d_op2_partial_diag, "factor_poisson_matrix_3d_op2_partial_diag", mesh->faces,
              op_arg_dat(mesh->order, -2, mesh->face2cells, 1, "int", OP_READ),
              op_arg_dat(mesh->faceNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(mesh->fmaskL,  -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(mesh->fmaskR,  -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(mesh->nx, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->ny, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->nz, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->fscale, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->sJ, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->rx, -2, mesh->face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->sx, -2, mesh->face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->tx, -2, mesh->face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->ry, -2, mesh->face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->sy, -2, mesh->face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->ty, -2, mesh->face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->rz, -2, mesh->face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->sz, -2, mesh->face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->tz, -2, mesh->face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mat_free_factor_copy, -2, mesh->face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(diag, 0, mesh->face2cells, DG_NP, DG_FP_STR, OP_INC),
              op_arg_dat(diag, 1, mesh->face2cells, DG_NP, DG_FP_STR, OP_INC));
  timer->endTimer("FactorPoissonMatrixFreeDiag3D - calc_op2");
}

void FactorPoissonMatrixFreeDiag3D::calc_opbc() {
  timer->startTimer("FactorPoissonMatrixFreeDiag3D - calc_opbc");
  if(mesh->bface2cells) {
    op_par_loop(factor_poisson_matrix_3d_bop_diag, "factor_poisson_matrix_3d_bop_diag", mesh->bfaces,
                op_arg_dat(mesh->order, 0, mesh->bface2cells, 1, "int", OP_READ),
                op_arg_dat(mesh->bfaceNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(bc_types, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mesh->bnx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bny, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bnz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bfscale, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bsJ, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->rx, 0, mesh->bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->sx, 0, mesh->bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->tx, 0, mesh->bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->ry, 0, mesh->bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->sy, 0, mesh->bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->ty, 0, mesh->bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->rz, 0, mesh->bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->sz, 0, mesh->bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->tz, 0, mesh->bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_factor_copy, 0, mesh->bface2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(diag, 0, mesh->bface2cells, DG_NP, DG_FP_STR, OP_INC));
  }
  timer->endTimer("FactorPoissonMatrixFreeDiag3D - calc_opbc");
}
