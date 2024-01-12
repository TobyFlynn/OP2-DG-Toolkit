#include "dg_matrices/2d/factor_poisson_matrix_free_block_diag_2d.h"

#include "op_seq.h"

#include "dg_constants/dg_constants.h"
#include "dg_op2_blas.h"

#include "timing.h"

extern DGConstants *constants;
extern Timing *timer;

FactorPoissonMatrixFreeBlockDiag2D::FactorPoissonMatrixFreeBlockDiag2D(DGMesh2D *m) : FactorPoissonMatrixFreeMult2D(m) {
  _mesh = m;

  block_diag = op_decl_dat(mesh->cells, DG_NP * DG_NP, DG_FP_STR, (DG_FP *)NULL, "poisson_matrix_free_block_diag");
}

void FactorPoissonMatrixFreeBlockDiag2D::set_bc_types(op_dat bc_ty) {
  bc_types = bc_ty;
  mat_free_set_bc_types(bc_ty);
}

void FactorPoissonMatrixFreeBlockDiag2D::set_factor(op_dat f) {
  factor = f;
  mat_free_set_factor(f);
}

void FactorPoissonMatrixFreeBlockDiag2D::apply_bc(op_dat rhs, op_dat bc) {
  mat_free_apply_bc(rhs, bc);
}

void FactorPoissonMatrixFreeBlockDiag2D::mult(op_dat in, op_dat out) {
  timer->startTimer("FactorPoissonMatrixFreeBlockDiag2D - mult");
  mat_free_mult(in, out);
  timer->endTimer("FactorPoissonMatrixFreeBlockDiag2D - mult");
}

void FactorPoissonMatrixFreeBlockDiag2D::mult_sp(op_dat in, op_dat out) {
  timer->startTimer("FactorPoissonMatrixFreeBlockDiag2D - mult sp");
  mat_free_mult_sp(in, out);
  timer->endTimer("FactorPoissonMatrixFreeBlockDiag2D - mult sp");
}

void FactorPoissonMatrixFreeBlockDiag2D::multJacobi_sp(op_dat in, op_dat out) {
  timer->startTimer("FactorPoissonMatrixFreeBlockDiag2D - multJacobi sp");
  mat_free_mult_sp(in, out);

  op_par_loop(poisson_block_diag_mult_jacobi_sp, "poisson_block_diag_mult_jacobi_sp", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(block_diag, -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(out, -1, OP_ID, DG_NP, "float", OP_RW));
  timer->endTimer("FactorPoissonMatrixFreeBlockDiag2D - multJacobi sp");
}

void FactorPoissonMatrixFreeBlockDiag2D::calc_mat_partial() {
  timer->startTimer("FactorPoissonMatrixFreeBlockDiag2D - calc_mat_partial");
  check_current_order();
  calc_op1();
  calc_op2();
  calc_opbc();
  petscMatResetRequired = true;
  timer->endTimer("FactorPoissonMatrixFreeBlockDiag2D - calc_mat_partial");
}

void FactorPoissonMatrixFreeBlockDiag2D::calc_op1() {
  timer->startTimer("FactorPoissonMatrixFreeBlockDiag2D - calc_op1");
  op_par_loop(factor_poisson_matrix_2d_op1, "factor_poisson_matrix_2d_op1", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->geof, -1, OP_ID, 5, DG_FP_STR, OP_READ),
              op_arg_dat(mat_free_factor_copy, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(block_diag, -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_WRITE));
  timer->endTimer("FactorPoissonMatrixFreeBlockDiag2D - calc_op1");
}

void FactorPoissonMatrixFreeBlockDiag2D::calc_op2() {
  timer->startTimer("FactorPoissonMatrixFreeBlockDiag2D - calc_op2");
  op_par_loop(factor_poisson_matrix_2d_op2_block_diag, "factor_poisson_matrix_2d_op2_block_diag", mesh->faces,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(mesh->reverse, -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(mesh->nx, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->ny, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->fscale, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->sJ, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->geof, -2, mesh->face2cells, 5, DG_FP_STR, OP_READ),
              op_arg_dat(mat_free_factor_copy, -2, mesh->face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(block_diag, 0, mesh->face2cells, DG_NP * DG_NP, DG_FP_STR, OP_INC),
              op_arg_dat(block_diag, 1, mesh->face2cells, DG_NP * DG_NP, DG_FP_STR, OP_INC));
  timer->endTimer("FactorPoissonMatrixFreeBlockDiag2D - calc_op2");
}

void FactorPoissonMatrixFreeBlockDiag2D::calc_opbc() {
  timer->startTimer("FactorPoissonMatrixFreeBlockDiag2D - calc_opbc");
  if(mesh->bface2cells) {
    op_par_loop(factor_poisson_matrix_2d_bop_block_diag, "factor_poisson_matrix_2d_bop_block_diag", mesh->bfaces,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_dat(mesh->bedgeNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(bc_types, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mesh->bnx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bny, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bfscale, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bsJ, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->geof, 0, mesh->bface2cells, 5, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_factor_copy, 0, mesh->bface2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(block_diag, 0, mesh->bface2cells, DG_NP * DG_NP, DG_FP_STR, OP_INC));
  }
  timer->endTimer("FactorPoissonMatrixFreeBlockDiag2D - calc_opbc");
}
