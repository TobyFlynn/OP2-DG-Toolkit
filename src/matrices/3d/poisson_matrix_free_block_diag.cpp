#include "dg_matrices/3d/poisson_matrix_free_block_diag_3d.h"

#include "op_seq.h"

#include "dg_constants/dg_constants.h"
#include "dg_op2_blas.h"

#include "timing.h"

extern DGConstants *constants;
extern Timing *timer;

PoissonMatrixFreeBlockDiag3D::PoissonMatrixFreeBlockDiag3D(DGMesh3D *m) : PoissonMatrixFreeMult3D(m) {
  _mesh = m;

  block_diag = op_decl_dat(mesh->cells, DG_NP * DG_NP, DG_FP_STR, (DG_FP *)NULL, "poisson_matrix_block_diag");
}

void PoissonMatrixFreeBlockDiag3D::set_bc_types(op_dat bc_ty) {
  bc_types = bc_ty;
  mat_free_set_bc_types(bc_ty);
}

void PoissonMatrixFreeBlockDiag3D::apply_bc(op_dat rhs, op_dat bc) {
  mat_free_apply_bc(rhs, bc);
}

void PoissonMatrixFreeBlockDiag3D::calc_mat_partial() {
  timer->startTimer("PoissonMatrixFreeBlockDiag3D - calc_mat_partial");
  calc_op1();
  calc_op2();
  calc_opbc();
  petscMatResetRequired = true;
  timer->endTimer("PoissonMatrixFreeBlockDiag3D - calc_mat_partial");
}

void PoissonMatrixFreeBlockDiag3D::mult(op_dat in, op_dat out) {
  timer->startTimer("PoissonMatrixFreeBlockDiag3D - mult");
  mat_free_mult(in, out);
  timer->endTimer("PoissonMatrixFreeBlockDiag3D - mult");
}

void PoissonMatrixFreeBlockDiag3D::calc_op1() {
  timer->startTimer("PoissonMatrixFreeBlockDiag3D - calc_op1");
  op_par_loop(poisson_matrix_3d_op1, "poisson_matrix_3d_op1", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
              op_arg_dat(block_diag, -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_WRITE));
  timer->endTimer("PoissonMatrixFreeBlockDiag3D - calc_op1");
}

void PoissonMatrixFreeBlockDiag3D::calc_op2() {
  timer->startTimer("PoissonMatrixFreeBlockDiag3D - calc_op2");
  // TODO full p-adaptivity
  op_par_loop(poisson_matrix_3d_op2_partial, "poisson_matrix_3d_op2_partial", mesh->faces,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->faceNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(mesh->fmaskL,  -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(mesh->fmaskR,  -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(mesh->nx, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->ny, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->nz, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->fscale, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->sJ, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->geof, -2, mesh->face2cells, 10, DG_FP_STR, OP_READ),
              op_arg_dat(block_diag, 0, mesh->face2cells, DG_NP * DG_NP, DG_FP_STR, OP_INC),
              op_arg_dat(block_diag, 1, mesh->face2cells, DG_NP * DG_NP, DG_FP_STR, OP_INC));
  timer->endTimer("PoissonMatrixFreeBlockDiag3D - calc_op2");
}

void PoissonMatrixFreeBlockDiag3D::calc_opbc() {
  timer->startTimer("PoissonMatrixFreeBlockDiag3D - calc_opbc");
  if(mesh->bface2cells) {
    op_par_loop(poisson_matrix_3d_bop, "poisson_matrix_3d_bop", mesh->bfaces,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_dat(mesh->bfaceNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(bc_types, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mesh->bnx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bny, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bnz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bfscale, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bsJ, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->geof, 0, mesh->bface2cells, 10, DG_FP_STR, OP_READ),
                op_arg_dat(block_diag, 0, mesh->bface2cells, DG_NP * DG_NP, DG_FP_STR, OP_INC));
  }
  timer->endTimer("PoissonMatrixFreeBlockDiag3D - calc_opbc");
}
