#include "dg_matrices/2d/factor_poisson_coarse_matrix_over_int_2d.h"

#include "op_seq.h"

#include "dg_op2_blas.h"
#include "dg_constants/dg_constants.h"

#include "timing.h"

extern Timing *timer;
extern DGConstants *constants;

FactorPoissonCoarseMatrixOverInt2D::FactorPoissonCoarseMatrixOverInt2D(DGMesh2D *m) : PoissonCoarseMatrixOverInt2D(m) {
  DG_FP *tmp_g_np = (DG_FP *)calloc(DG_G_NP * mesh->cells->size, sizeof(DG_FP));
  gFactor = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, tmp_g_np, "poisson_gFactor");
  free(tmp_g_np);
}

void FactorPoissonCoarseMatrixOverInt2D::set_factor(op_dat f) {
  factor = f;
}

void FactorPoissonCoarseMatrixOverInt2D::calc_op1() {
  timer->startTimer("FactorPoissonCoarseMatrixOverInt2D - calc_op1");
  op_par_loop(factor_poisson_coarse_matrix_2d_op1_over_int, "factor_poisson_coarse_matrix_2d_op1_over_int", mesh->cells,
              op_arg_gbl(constants->get_mat_ptr(DGConstants::DR), DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::DS), DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::MASS), DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->rx, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->sx, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->ry, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->sy, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->J,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(factor,   -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op1, -1, OP_ID, DG_NP_N1 * DG_NP_N1, DG_FP_STR, OP_WRITE));
  timer->endTimer("FactorPoissonCoarseMatrixOverInt2D - calc_op1");
}

void FactorPoissonCoarseMatrixOverInt2D::calc_op2() {
  timer->startTimer("FactorPoissonCoarseMatrixOverInt2D - calc_op2");
  op2_gemv(mesh, false, 1.0, DGConstants::GAUSS_INTERP, factor, 0.0, gFactor);

  op_par_loop(factor_poisson_coarse_matrix_2d_op2_over_int, "factor_poisson_coarse_matrix_2d_op2_over_int", mesh->faces,
              op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F0DR), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F0DS), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F1DR), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F1DS), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F2DR), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F2DS), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_FINTERP0), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_FINTERP1), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_FINTERP2), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(mesh->reverse, -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(mesh->x, -2, mesh->face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->y, -2, mesh->face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->gauss->sJ, -2, mesh->face2cells, DG_G_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->gauss->nx, -2, mesh->face2cells, DG_G_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->gauss->ny, -2, mesh->face2cells, DG_G_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->fscale_c, -2, mesh->face2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
              op_arg_dat(gFactor, -2, mesh->face2cells, DG_G_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op1, 0, mesh->face2cells, DG_NP_N1 * DG_NP_N1, DG_FP_STR, OP_INC),
              op_arg_dat(op1, 1, mesh->face2cells, DG_NP_N1 * DG_NP_N1, DG_FP_STR, OP_INC),
              op_arg_dat(op2[0], -1, OP_ID, DG_NP_N1 * DG_NP_N1, DG_FP_STR, OP_WRITE),
              op_arg_dat(op2[1], -1, OP_ID, DG_NP_N1 * DG_NP_N1, DG_FP_STR, OP_WRITE));
  timer->endTimer("FactorPoissonCoarseMatrixOverInt2D - calc_op2");
}

void FactorPoissonCoarseMatrixOverInt2D::calc_opbc() {
  timer->startTimer("FactorPoissonCoarseMatrixOverInt2D - calc_opbc");
  if(mesh->bface2cells) {
    op_par_loop(factor_poisson_coarse_matrix_2d_bop_over_int, "factor_poisson_coarse_matrix_2d_bop_over_int", mesh->bfaces,
                op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F0DR), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F0DS), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F1DR), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F1DS), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F2DR), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F2DS), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_FINTERP0), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_FINTERP1), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_FINTERP2), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(bc_types, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mesh->bedgeNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mesh->x, 0, mesh->bface2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->y, 0, mesh->bface2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->gauss->sJ, 0, mesh->bface2cells, DG_G_NP, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->gauss->nx, 0, mesh->bface2cells, DG_G_NP, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->gauss->ny, 0, mesh->bface2cells, DG_G_NP, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->fscale_c, 0, mesh->bface2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
                op_arg_dat(gFactor, 0, mesh->bface2cells, DG_G_NP, DG_FP_STR, OP_READ),
                op_arg_dat(op1, 0, mesh->bface2cells, DG_NP_N1 * DG_NP_N1, DG_FP_STR, OP_INC),
                op_arg_dat(opbc, -1, OP_ID, DG_GF_NP * DG_NP_N1, DG_FP_STR, OP_WRITE));
  }
  timer->endTimer("FactorPoissonCoarseMatrixOverInt2D - calc_opbc");
}
