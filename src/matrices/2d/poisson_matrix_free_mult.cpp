#include "dg_matrices/2d/poisson_matrix_free_mult_2d.h"

#include "op_seq.h"

#include "dg_constants/dg_constants.h"
#include "dg_op2_blas.h"
#include "dg_dat_pool.h"

#include "timing.h"

extern DGConstants *constants;
extern Timing *timer;
extern DGDatPool *dg_dat_pool;

PoissonMatrixFreeMult2D::PoissonMatrixFreeMult2D(DGMesh2D *m) {
  mesh = m;

  mat_free_tau_c = op_decl_dat(mesh->cells, 3, DG_FP_STR, (DG_FP *)NULL, "mat_free_tau_c");
  mat_free_tau_c_sp = op_decl_dat(mesh->cells, 3, "float", (float *)NULL, "mat_free_tau_c_sp");
}

void PoissonMatrixFreeMult2D::mat_free_set_bc_types(op_dat bc_ty) {
  mat_free_bcs = bc_ty;
}

void PoissonMatrixFreeMult2D::mat_free_apply_bc(op_dat rhs, op_dat bc) {
  if(mesh->bface2cells) {
    op_par_loop(pmf_2d_apply_bc, "pmf_2d_apply_bc", mesh->bfaces,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_dat(mesh->bedgeNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mat_free_bcs, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mesh->bnx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bny, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bfscale, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bsJ, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->geof, 0, mesh->bface2cells, 5, DG_FP_STR, OP_READ),
                op_arg_dat(bc, -1, OP_ID, DG_NPF, DG_FP_STR, OP_READ),
                op_arg_dat(rhs, 0, mesh->bface2cells, DG_NP, DG_FP_STR, OP_INC));
  }
}

void PoissonMatrixFreeMult2D::mat_free_pre_compute_tau() {
  timer->startTimer("PoissonMatrixFreeMult3D - calc tau");
  op_par_loop(pmf_2d_calc_tau_faces, "pmf_2d_calc_tau_faces", mesh->faces,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(mesh->fscale, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mat_free_tau_c, -2, mesh->face2cells, 3, DG_FP_STR, OP_WRITE));
  if(mesh->bface2cells) {
    op_par_loop(pmf_2d_calc_tau_bfaces, "pmf_2d_calc_tau_bfaces", mesh->bfaces,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_dat(mesh->bedgeNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mesh->bfscale, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_tau_c, 0, mesh->bface2cells, 3, DG_FP_STR, OP_WRITE));
  }

  op_par_loop(pmf_2d_calc_tau_faces_sp, "pmf_2d_calc_tau_faces_sp", mesh->faces,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(mesh->fscale, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mat_free_tau_c_sp, -2, mesh->face2cells, 3, "float", OP_WRITE));
  if(mesh->bface2cells) {
    op_par_loop(pmf_2d_calc_tau_bfaces_sp, "pmf_2d_calc_tau_bfaces_sp", mesh->bfaces,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_dat(mesh->bedgeNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mesh->bfscale, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_tau_c_sp, 0, mesh->bface2cells, 3, "float", OP_WRITE));
  }
  timer->endTimer("PoissonMatrixFreeMult3D - calc tau");
}

void PoissonMatrixFreeMult2D::mat_free_mult(op_dat in, op_dat out) {
  mat_free_pre_compute_tau();
  timer->startTimer("PoissonMatrixFreeMult2D - mult");
  DGTempDat tmp_grad0 = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_grad1 = dg_dat_pool->requestTempDatCells(DG_NP);
  timer->startTimer("PoissonMatrixFreeMult2D - mult grad");
  mesh->grad(in, tmp_grad0.dat, tmp_grad1.dat);
  timer->endTimer("PoissonMatrixFreeMult2D - mult grad");

  DGTempDat tmp_npf0 = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);
  DGTempDat tmp_npf1 = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);
  DGTempDat tmp_npf2 = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);

  op_par_loop(zero_npf_3_tk, "zero_npf_3_tk", mesh->cells,
              op_arg_dat(tmp_npf0.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(tmp_npf1.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(tmp_npf2.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));

  timer->startTimer("PoissonMatrixFreeMult2D - mult faces");
  mesh->jump(in, tmp_npf0.dat);
  mesh->avg(tmp_grad0.dat, tmp_npf1.dat);
  mesh->avg(tmp_grad1.dat, tmp_npf2.dat);
  timer->endTimer("PoissonMatrixFreeMult2D - mult faces");

  timer->startTimer("PoissonMatrixFreeMult2D - mult bfaces");
  if(mesh->bface2cells) {
    op_par_loop(pmf_2d_mult_avg_jump, "pmf_2d_mult_avg_jump", mesh->bfaces,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_dat(mat_free_bcs, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mesh->bedgeNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(in, 0, mesh->bface2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(tmp_grad0.dat, 0, mesh->bface2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(tmp_grad1.dat, 0, mesh->bface2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(tmp_npf0.dat, 0, mesh->bface2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC),
                op_arg_dat(tmp_npf1.dat, 0, mesh->bface2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC),
                op_arg_dat(tmp_npf2.dat, 0, mesh->bface2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC));
  }
  timer->endTimer("PoissonMatrixFreeMult2D - mult bfaces");

  timer->startTimer("PoissonMatrixFreeMult2D - finish flux");
  op_par_loop(pmf_2d_mult_flux, "pmf_2d_mult_flux", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->nx_c_new, -1, OP_ID, 3, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->ny_c_new, -1, OP_ID, 3, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->sJ_c_new, -1, OP_ID, 3, DG_FP_STR, OP_READ),
              op_arg_dat(mat_free_tau_c, -1, OP_ID, 3, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_npf0.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_RW),
              op_arg_dat(tmp_npf1.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_RW),
              op_arg_dat(tmp_npf2.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_RW));
  timer->endTimer("PoissonMatrixFreeMult2D - finish flux");

  timer->startTimer("PoissonMatrixFreeMult2D - mult cells");
  timer->startTimer("PoissonMatrixFreeMult2D - mult cells MM");
  mesh->mass(tmp_grad0.dat);
  mesh->mass(tmp_grad1.dat);
  timer->endTimer("PoissonMatrixFreeMult2D - mult cells MM");

  timer->startTimer("PoissonMatrixFreeMult2D - mult cells Emat");
  op2_gemv(mesh, false, 1.0, DGConstants::EMAT, tmp_npf1.dat, 1.0, tmp_grad0.dat);
  op2_gemv(mesh, false, 1.0, DGConstants::EMAT, tmp_npf2.dat, 1.0, tmp_grad1.dat);
  op2_gemv(mesh, false, 1.0, DGConstants::EMAT, tmp_npf0.dat, 0.0, out);
  timer->endTimer("PoissonMatrixFreeMult2D - mult cells Emat");

  timer->startTimer("PoissonMatrixFreeMult2D - mult cells cells");
  op_par_loop(pmf_2d_mult_cells_geof, "pmf_2d_mult_cells_geof", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->geof, -1, OP_ID, 5, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_grad0.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(tmp_grad1.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));

  op2_gemv(mesh, true, 1.0, DGConstants::DR, tmp_grad0.dat, 1.0, out);
  op2_gemv(mesh, true, 1.0, DGConstants::DS, tmp_grad1.dat, 1.0, out);
  timer->endTimer("PoissonMatrixFreeMult2D - mult cells cells");
  timer->endTimer("PoissonMatrixFreeMult2D - mult cells");
  dg_dat_pool->releaseTempDatCells(tmp_grad0);
  dg_dat_pool->releaseTempDatCells(tmp_grad1);
  dg_dat_pool->releaseTempDatCells(tmp_npf0);
  dg_dat_pool->releaseTempDatCells(tmp_npf1);
  dg_dat_pool->releaseTempDatCells(tmp_npf2);
  timer->endTimer("PoissonMatrixFreeMult2D - mult");
}

void PoissonMatrixFreeMult2D::mat_free_mult_sp(op_dat in, op_dat out) {
  mat_free_pre_compute_tau();
  timer->startTimer("PoissonMatrixFreeMult2D - mult");
  DGTempDat tmp_grad0 = dg_dat_pool->requestTempDatCellsSP(DG_NP);
  DGTempDat tmp_grad1 = dg_dat_pool->requestTempDatCellsSP(DG_NP);
  timer->startTimer("PoissonMatrixFreeMult2D - mult grad");
  DGTempDat tmp_r = dg_dat_pool->requestTempDatCellsSP(DG_NP);
  DGTempDat tmp_s = dg_dat_pool->requestTempDatCellsSP(DG_NP);
  op2_gemv_sp(mesh, false, 1.0, DGConstants::DR, in, 0.0, tmp_r.dat);
  op2_gemv_sp(mesh, false, 1.0, DGConstants::DS, in, 0.0, tmp_s.dat);
  op_par_loop(grad_2d_sp, "grad_2d_sp", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->geof, -1, OP_ID, 5, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_r.dat, -1, OP_ID, DG_NP, "float", OP_READ),
              op_arg_dat(tmp_s.dat, -1, OP_ID, DG_NP, "float", OP_READ),
              op_arg_dat(tmp_grad0.dat, -1, OP_ID, DG_NP, "float", OP_WRITE),
              op_arg_dat(tmp_grad1.dat, -1, OP_ID, DG_NP, "float", OP_WRITE));
  dg_dat_pool->releaseTempDatCellsSP(tmp_r);
  dg_dat_pool->releaseTempDatCellsSP(tmp_s);
  timer->endTimer("PoissonMatrixFreeMult2D - mult grad");

  DGTempDat tmp_npf0 = dg_dat_pool->requestTempDatCellsSP(DG_NUM_FACES * DG_NPF);
  DGTempDat tmp_npf1 = dg_dat_pool->requestTempDatCellsSP(DG_NUM_FACES * DG_NPF);
  DGTempDat tmp_npf2 = dg_dat_pool->requestTempDatCellsSP(DG_NUM_FACES * DG_NPF);

  op_par_loop(zero_npf_3_sp, "zero_npf_3_sp", mesh->cells,
              op_arg_dat(tmp_npf0.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_WRITE),
              op_arg_dat(tmp_npf1.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_WRITE),
              op_arg_dat(tmp_npf2.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_WRITE));

  timer->startTimer("PoissonMatrixFreeMult2D - mult faces");
  mesh->jump_sp(in, tmp_npf0.dat);
  mesh->avg_sp(tmp_grad0.dat, tmp_npf1.dat);
  mesh->avg_sp(tmp_grad1.dat, tmp_npf2.dat);
  timer->endTimer("PoissonMatrixFreeMult2D - mult faces");

  timer->startTimer("PoissonMatrixFreeMult2D - mult bfaces");
  if(mesh->bface2cells) {
    op_par_loop(pmf_2d_mult_avg_jump_sp, "pmf_2d_mult_avg_jump_sp", mesh->bfaces,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_dat(mat_free_bcs, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mesh->bedgeNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(in, 0, mesh->bface2cells, DG_NP, "float", OP_READ),
                op_arg_dat(tmp_grad0.dat, 0, mesh->bface2cells, DG_NP, "float", OP_READ),
                op_arg_dat(tmp_grad1.dat, 0, mesh->bface2cells, DG_NP, "float", OP_READ),
                op_arg_dat(tmp_npf0.dat, 0, mesh->bface2cells, DG_NUM_FACES * DG_NPF, "float", OP_INC),
                op_arg_dat(tmp_npf1.dat, 0, mesh->bface2cells, DG_NUM_FACES * DG_NPF, "float", OP_INC),
                op_arg_dat(tmp_npf2.dat, 0, mesh->bface2cells, DG_NUM_FACES * DG_NPF, "float", OP_INC));
  }
  timer->endTimer("PoissonMatrixFreeMult2D - mult bfaces");

  timer->startTimer("PoissonMatrixFreeMult2D - finish flux");
  op_par_loop(pmf_2d_mult_flux_sp, "pmf_2d_mult_flux_sp", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->nx_c_new, -1, OP_ID, 3, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->ny_c_new, -1, OP_ID, 3, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->sJ_c_new, -1, OP_ID, 3, DG_FP_STR, OP_READ),
              op_arg_dat(mat_free_tau_c_sp, -1, OP_ID, 3, "float", OP_READ),
              op_arg_dat(tmp_npf0.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_RW),
              op_arg_dat(tmp_npf1.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_RW),
              op_arg_dat(tmp_npf2.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_RW));
  timer->endTimer("PoissonMatrixFreeMult2D - finish flux");

  timer->startTimer("PoissonMatrixFreeMult2D - mult cells");
  timer->startTimer("PoissonMatrixFreeMult2D - mult cells MM");
  mesh->mass_sp(tmp_grad0.dat);
  mesh->mass_sp(tmp_grad1.dat);
  timer->endTimer("PoissonMatrixFreeMult2D - mult cells MM");

  timer->startTimer("PoissonMatrixFreeMult2D - mult cells Emat");
  op2_gemv_sp(mesh, false, 1.0, DGConstants::EMAT, tmp_npf1.dat, 1.0, tmp_grad0.dat);
  op2_gemv_sp(mesh, false, 1.0, DGConstants::EMAT, tmp_npf2.dat, 1.0, tmp_grad1.dat);
  op2_gemv_sp(mesh, false, 1.0, DGConstants::EMAT, tmp_npf0.dat, 0.0, out);
  timer->endTimer("PoissonMatrixFreeMult2D - mult cells Emat");

  timer->startTimer("PoissonMatrixFreeMult2D - mult cells cells");
  op_par_loop(pmf_2d_mult_cells_geof_sp, "pmf_2d_mult_cells_geof_sp", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->geof, -1, OP_ID, 5, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_grad0.dat, -1, OP_ID, DG_NP, "float", OP_RW),
              op_arg_dat(tmp_grad1.dat, -1, OP_ID, DG_NP, "float", OP_RW));

  op2_gemv_sp(mesh, true, 1.0, DGConstants::DR, tmp_grad0.dat, 1.0, out);
  op2_gemv_sp(mesh, true, 1.0, DGConstants::DS, tmp_grad1.dat, 1.0, out);
  timer->endTimer("PoissonMatrixFreeMult2D - mult cells cells");
  timer->endTimer("PoissonMatrixFreeMult2D - mult cells");
  dg_dat_pool->releaseTempDatCellsSP(tmp_grad0);
  dg_dat_pool->releaseTempDatCellsSP(tmp_grad1);
  dg_dat_pool->releaseTempDatCellsSP(tmp_npf0);
  dg_dat_pool->releaseTempDatCellsSP(tmp_npf1);
  dg_dat_pool->releaseTempDatCellsSP(tmp_npf2);
  timer->endTimer("PoissonMatrixFreeMult2D - mult");
}
