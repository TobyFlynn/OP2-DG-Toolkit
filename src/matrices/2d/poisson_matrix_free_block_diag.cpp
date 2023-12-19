#include "dg_matrices/2d/poisson_matrix_free_block_diag_2d.h"

#include "op_seq.h"

#include "dg_constants/dg_constants.h"
#include "dg_op2_blas.h"

#include "timing.h"

extern DGConstants *constants;
extern Timing *timer;

PoissonMatrixFreeBlockDiag2D::PoissonMatrixFreeBlockDiag2D(DGMesh2D *m) : PoissonMatrixFreeMult2D(m) {
  _mesh = m;

  block_diag = op_decl_dat(mesh->cells, DG_NP * DG_NP, DG_FP_STR, (DG_FP *)NULL, "poisson_matrix_free_block_diag");
}

void PoissonMatrixFreeBlockDiag2D::set_bc_types(op_dat bc_ty) {
  bc_types = bc_ty;
  mat_free_set_bc_types(bc_ty);
}

void PoissonMatrixFreeBlockDiag2D::apply_bc(op_dat rhs, op_dat bc) {
  mat_free_apply_bc(rhs, bc);
}

void PoissonMatrixFreeBlockDiag2D::calc_mat_partial() {
  timer->startTimer("PoissonMatrixFreeBlockDiag2D - calc_mat_partial");
  calc_op1();
  calc_op2();
  calc_opbc();
  petscMatResetRequired = true;
  timer->endTimer("PoissonMatrixFreeBlockDiag2D - calc_mat_partial");
}

void PoissonMatrixFreeBlockDiag2D::mult(op_dat in, op_dat out) {
  timer->startTimer("PoissonMatrixFreeBlockDiag2D - mult");
  mat_free_mult(in, out);
  timer->endTimer("PoissonMatrixFreeBlockDiag2D - mult");
}

void PoissonMatrixFreeBlockDiag2D::mult_sp(op_dat in, op_dat out) {
  timer->startTimer("PoissonMatrixFreeBlockDiag2D - mult sp");
  mat_free_mult_sp(in, out);
  timer->endTimer("PoissonMatrixFreeBlockDiag2D - mult sp");
}

void PoissonMatrixFreeBlockDiag2D::multJacobi_sp(op_dat in, op_dat out) {
  timer->startTimer("PoissonMatrixFreeBlockDiag2D - multJacobi sp");
  mat_free_mult_sp(in, out);

  op_par_loop(poisson_block_diag_mult_jacobi_sp, "poisson_block_diag_mult_jacobi_sp", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(block_diag, -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(out, -1, OP_ID, DG_NP, "float", OP_RW));
  timer->endTimer("PoissonMatrixFreeBlockDiag2D - multJacobi sp");
}

void PoissonMatrixFreeBlockDiag2D::calc_op1() {
  timer->startTimer("PoissonMatrixFreeBlockDiag2D - calc_op1");
  op_par_loop(poisson_matrix_2d_op1, "poisson_matrix_2d_op1", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->geof, -1, OP_ID, 5, DG_FP_STR, OP_READ),
              op_arg_dat(block_diag, -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_WRITE));
  timer->endTimer("PoissonMatrixFreeBlockDiag2D - calc_op1");
}

void PoissonMatrixFreeBlockDiag2D::calc_op2() {
  timer->startTimer("PoissonMatrixFreeBlockDiag2D - calc_op2");
  op_par_loop(poisson_matrix_2d_op2_block_diag, "poisson_matrix_2d_op2_block_diag", mesh->faces,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(mesh->reverse, -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(mesh->nx, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->ny, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->fscale, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->sJ, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->geof, -2, mesh->face2cells, 5, DG_FP_STR, OP_READ),
              op_arg_dat(block_diag, 0, mesh->face2cells, DG_NP * DG_NP, DG_FP_STR, OP_INC),
              op_arg_dat(block_diag, 1, mesh->face2cells, DG_NP * DG_NP, DG_FP_STR, OP_INC));
  timer->endTimer("PoissonMatrixFreeBlockDiag2D - calc_op2");
}

void PoissonMatrixFreeBlockDiag2D::calc_opbc() {
  timer->startTimer("PoissonMatrixFreeBlockDiag2D - calc_opbc");
  if(mesh->bface2cells) {
    op_par_loop(poisson_matrix_2d_bop_block_diag, "poisson_matrix_2d_bop_block_diag", mesh->bfaces,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_dat(mesh->bedgeNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(bc_types, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mesh->bnx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bny, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bfscale, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bsJ, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->geof, 0, mesh->bface2cells, 5, DG_FP_STR, OP_READ),
                op_arg_dat(block_diag, 0, mesh->bface2cells, DG_NP * DG_NP, DG_FP_STR, OP_INC));
  }
  timer->endTimer("PoissonMatrixFreeBlockDiag2D - calc_opbc");
}
