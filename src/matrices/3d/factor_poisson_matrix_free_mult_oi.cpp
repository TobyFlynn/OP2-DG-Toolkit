#include "dg_matrices/3d/factor_poisson_matrix_free_mult_oi_3d.h"

#include "op_seq.h"

#include "dg_constants/dg_constants.h"
#include "dg_op2_blas.h"
#include "dg_dat_pool.h"

#include "timing.h"

extern DGConstants *constants;
extern Timing *timer;
extern DGDatPool *dg_dat_pool;

FactorPoissonMatrixFreeMultOI3D::FactorPoissonMatrixFreeMultOI3D(DGMesh3D *m) : PoissonMatrixFreeMult3D(m) {
  mat_free_gtau = op_decl_dat(mesh->cells, 4, DG_FP_STR, (DG_FP *)NULL, "poisson_matrix_free_tau");
  mat_free_factor_copy = op_decl_dat(mesh->cells, DG_NP, DG_FP_STR, (DG_FP *)NULL, "poisson_matrix_free_factor_copy");
  mat_free_factor_copy_sp = op_decl_dat(mesh->cells, DG_NP, "float", (float *)NULL, "poisson_matrix_free_factor_copy_sp");
}

void FactorPoissonMatrixFreeMultOI3D::mat_free_set_factor_oi(op_dat f) {
  mat_free_factor_oi = f;
}

void FactorPoissonMatrixFreeMultOI3D::mat_free_set_factor_surf_oi(op_dat f) {
  mat_free_factor_surf_oi = f;
}

void FactorPoissonMatrixFreeMultOI3D::calc_tau() {
  timer->startTimer("FactorPoissonMatrixFreeMultOI3D - calc tau");
  op_par_loop(fpmf_3d_calc_tau_faces, "fpmf_3d_calc_tau_faces", mesh->faces,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->faceNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(mesh->fmaskR, -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(mesh->fscale, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(mat_free_factor_copy, -2, mesh->face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mat_free_tau_c, -2, mesh->face2cells, 4, DG_FP_STR, OP_WRITE));
  if(mesh->bface2cells) {
    op_par_loop(fpmf_3d_calc_tau_bfaces, "fpmf_3d_calc_tau_bfaces", mesh->bfaces,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_dat(mesh->bfaceNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mesh->bfscale, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_factor_copy, 0, mesh->bface2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_tau_c, 0, mesh->bface2cells, 4, DG_FP_STR, OP_WRITE));
  }
  timer->endTimer("FactorPoissonMatrixFreeMultOI3D - calc tau");
}

void FactorPoissonMatrixFreeMultOI3D::mat_free_set_factor(op_dat f) {
  mat_free_factor = f;

  factor_order = mesh->order_int;
  current_order = mesh->order_int;

  op_par_loop(copy_dg_np_tk, "copy_dg_np_tk", mesh->cells,
              op_arg_dat(mat_free_factor, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mat_free_factor_copy, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  op_par_loop(copy_dg_np_dp2sp_tk, "copy_dg_np_tk", mesh->cells,
              op_arg_dat(mat_free_factor, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mat_free_factor_copy_sp, -1, OP_ID, DG_NP, "float", OP_WRITE));

  calc_tau();
}

void FactorPoissonMatrixFreeMultOI3D::check_current_order() {
  timer->startTimer("FactorPoissonMatrixFreeMultOI3D - check order");
  if(current_order != mesh->order_int) {
    mesh->interp_dat_between_orders(current_order, mesh->order_int, mat_free_factor, mat_free_factor_copy);
    current_order = mesh->order_int;

    op_par_loop(copy_dg_np_dp2sp_tk, "copy_dg_np_tk", mesh->cells,
                op_arg_dat(mat_free_factor, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_factor_copy_sp, -1, OP_ID, DG_NP, "float", OP_WRITE));

    calc_tau();
  }
  timer->endTimer("FactorPoissonMatrixFreeMultOI3D - check order");
}

void FactorPoissonMatrixFreeMultOI3D::mat_free_apply_bc(op_dat rhs, op_dat bc) {
  if(mesh->bface2cells) {
    check_current_order();
    op_par_loop(fpmf_3d_apply_bc, "fpmf_3d_apply_bc", mesh->bfaces,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_dat(mesh->bfaceNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mat_free_bcs, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mesh->bnx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bny, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bnz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bfscale, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->bsJ, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->geof, 0, mesh->bface2cells, 10, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_factor_copy, 0, mesh->bface2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(bc, -1, OP_ID, DG_NPF, DG_FP_STR, OP_READ),
                op_arg_dat(rhs, 0, mesh->bface2cells, DG_NP, DG_FP_STR, OP_INC));
  }
}

void FactorPoissonMatrixFreeMultOI3D::mat_free_mult(op_dat in, op_dat out) {
  check_current_order();
  timer->startTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " mult");
  DGTempDat tmp_grad0 = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_grad1 = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_grad2 = dg_dat_pool->requestTempDatCells(DG_NP);
  timer->startTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " mult grad");
  op2_gemv_halo_exchange(mesh, false, 1.0, DGConstants::DR, in, 0.0, tmp_grad0.dat);
  op2_gemv_halo_exchange(mesh, false, 1.0, DGConstants::DS, in, 0.0, tmp_grad1.dat);
  op2_gemv_halo_exchange(mesh, false, 1.0, DGConstants::DT, in, 0.0, tmp_grad2.dat);
  if(mesh->order_int == DG_ORDER){
    // TODO halo compute for this
    DGTempDat tmp_grad0_oi = dg_dat_pool->requestTempDatCells(DG_CUB_3D_NP);
    DGTempDat tmp_grad1_oi = dg_dat_pool->requestTempDatCells(DG_CUB_3D_NP);
    DGTempDat tmp_grad2_oi = dg_dat_pool->requestTempDatCells(DG_CUB_3D_NP);

    op2_gemv(mesh, false, 1.0, DGConstants::CUB3D_INTERP, tmp_grad0.dat, 0.0, tmp_grad0_oi.dat);
    op2_gemv(mesh, false, 1.0, DGConstants::CUB3D_INTERP, tmp_grad1.dat, 0.0, tmp_grad1_oi.dat);
    op2_gemv(mesh, false, 1.0, DGConstants::CUB3D_INTERP, tmp_grad2.dat, 0.0, tmp_grad2_oi.dat);

    op_par_loop(fpmf_3d_grad_oi, "fpmf_3d_grad_oi", mesh->cells,
                op_arg_dat(mesh->geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_factor_oi, -1, OP_ID, DG_CUB_3D_NP, DG_FP_STR, OP_READ),
                op_arg_dat(tmp_grad0_oi.dat, -1, OP_ID, DG_CUB_3D_NP, DG_FP_STR, OP_RW),
                op_arg_dat(tmp_grad1_oi.dat, -1, OP_ID, DG_CUB_3D_NP, DG_FP_STR, OP_RW),
                op_arg_dat(tmp_grad2_oi.dat, -1, OP_ID, DG_CUB_3D_NP, DG_FP_STR, OP_RW));

    op2_gemv(mesh, false, 1.0, DGConstants::CUB3D_PROJ, tmp_grad0_oi.dat, 0.0, tmp_grad0.dat);
    op2_gemv(mesh, false, 1.0, DGConstants::CUB3D_PROJ, tmp_grad1_oi.dat, 0.0, tmp_grad1.dat);
    op2_gemv(mesh, false, 1.0, DGConstants::CUB3D_PROJ, tmp_grad2_oi.dat, 0.0, tmp_grad2.dat);

    dg_dat_pool->releaseTempDatCells(tmp_grad0_oi);
    dg_dat_pool->releaseTempDatCells(tmp_grad1_oi);
    dg_dat_pool->releaseTempDatCells(tmp_grad2_oi);
  } else {
    op_par_loop(fpmf_3d_grad_2, "fpmf_3d_grad_2:force_halo_compute", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
              op_arg_dat(mat_free_factor_copy, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_grad0.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(tmp_grad1.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(tmp_grad2.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  }
  timer->endTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " mult grad");

  DGTempDat tmp_npf0 = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);
  DGTempDat tmp_npf1 = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);
  DGTempDat tmp_npf2 = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);
  DGTempDat tmp_npf3 = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);

  op_par_loop(zero_npf, "zero_npf", mesh->cells,
              op_arg_dat(tmp_npf0.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));

  op_par_loop(zero_npf_3_tk, "zero_npf_3_tk", mesh->cells,
              op_arg_dat(tmp_npf1.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(tmp_npf2.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(tmp_npf3.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));

  timer->startTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " mult faces");
  mesh->jump(in, tmp_npf0.dat);
  mesh->avg(tmp_grad0.dat, tmp_npf1.dat);
  mesh->avg(tmp_grad1.dat, tmp_npf2.dat);
  mesh->avg(tmp_grad2.dat, tmp_npf3.dat);
  timer->endTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " mult faces");

  timer->startTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " mult bfaces");
  if(mesh->bface2cells) {
    op_par_loop(pmf_3d_mult_avg_jump, "pmf_3d_mult_avg_jump", mesh->bfaces,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_dat(mat_free_bcs, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mesh->bfaceNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(in, 0, mesh->bface2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(tmp_grad0.dat, 0, mesh->bface2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(tmp_grad1.dat, 0, mesh->bface2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(tmp_grad2.dat, 0, mesh->bface2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(tmp_npf0.dat, 0, mesh->bface2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC),
                op_arg_dat(tmp_npf1.dat, 0, mesh->bface2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC),
                op_arg_dat(tmp_npf2.dat, 0, mesh->bface2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC),
                op_arg_dat(tmp_npf3.dat, 0, mesh->bface2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC));
  }
  timer->endTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " mult bfaces");

  if(mesh->order_int == DG_ORDER) {
    DGTempDat jump_cub = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_CUB_SURF_3D_NP);
    DGTempDat fluxX_cub = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_CUB_SURF_3D_NP);
    DGTempDat fluxY_cub = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_CUB_SURF_3D_NP);
    DGTempDat fluxZ_cub = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_CUB_SURF_3D_NP);
    op2_gemv(mesh, false, 1.0, DGConstants::CUBSURF3D_INTERP, tmp_npf0.dat, 0.0, jump_cub.dat);
    op_par_loop(fpmf_3d_mult_flux_oi_0, "fpmf_3d_mult_flux_oi_0", mesh->cells,
                op_arg_dat(mesh->nx_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->ny_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->nz_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->sJ_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_factor_surf_oi, -1, OP_ID, DG_NUM_FACES * DG_CUB_SURF_3D_NP, DG_FP_STR, OP_READ),
                op_arg_dat(jump_cub.dat,  -1, OP_ID, DG_NUM_FACES * DG_CUB_SURF_3D_NP, DG_FP_STR, OP_READ),
                op_arg_dat(fluxX_cub.dat, -1, OP_ID, DG_NUM_FACES * DG_CUB_SURF_3D_NP, DG_FP_STR, OP_WRITE),
                op_arg_dat(fluxY_cub.dat, -1, OP_ID, DG_NUM_FACES * DG_CUB_SURF_3D_NP, DG_FP_STR, OP_WRITE),
                op_arg_dat(fluxZ_cub.dat, -1, OP_ID, DG_NUM_FACES * DG_CUB_SURF_3D_NP, DG_FP_STR, OP_WRITE));
    op2_gemv(mesh, false, 1.0, DGConstants::CUBSURF3D_LIFT, fluxX_cub.dat, 1.0, tmp_grad0.dat);
    op2_gemv(mesh, false, 1.0, DGConstants::CUBSURF3D_LIFT, fluxY_cub.dat, 1.0, tmp_grad1.dat);
    op2_gemv(mesh, false, 1.0, DGConstants::CUBSURF3D_LIFT, fluxZ_cub.dat, 1.0, tmp_grad2.dat);
    dg_dat_pool->releaseTempDatCells(jump_cub);
    dg_dat_pool->releaseTempDatCells(fluxX_cub);
    dg_dat_pool->releaseTempDatCells(fluxY_cub);
    dg_dat_pool->releaseTempDatCells(fluxZ_cub);
    op_par_loop(fpmf_3d_mult_flux_oi_1, "fpmf_3d_mult_flux_oi_1", mesh->cells,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_dat(mesh->nx_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->ny_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->nz_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->sJ_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_tau_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(tmp_npf0.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_RW),
                op_arg_dat(tmp_npf1.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_RW),
                op_arg_dat(tmp_npf2.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_RW),
                op_arg_dat(tmp_npf3.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_RW));
    mesh->mass(tmp_grad0.dat);
    mesh->mass(tmp_grad1.dat);
    mesh->mass(tmp_grad2.dat);
    op2_gemv(mesh, false, 1.0, DGConstants::EMAT, tmp_npf0.dat, 0.0, out);
  } else {
    timer->startTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " finish flux");
    op_par_loop(fpmf_3d_mult_flux, "fpmf_3d_mult_flux", mesh->cells,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_dat(mesh->nx_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->ny_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->nz_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->sJ_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_tau_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_factor_copy, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(tmp_npf0.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_RW),
                op_arg_dat(tmp_npf1.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_RW),
                op_arg_dat(tmp_npf2.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_RW),
                op_arg_dat(tmp_npf3.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_RW));
    timer->endTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " finish flux");

    timer->startTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " mult cells MM");
    mesh->mass(tmp_grad0.dat);
    mesh->mass(tmp_grad1.dat);
    mesh->mass(tmp_grad2.dat);
    timer->endTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " mult cells MM");

    timer->startTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " mult cells Emat");
    op2_gemv(mesh, false, 1.0, DGConstants::EMAT, tmp_npf1.dat, 1.0, tmp_grad0.dat);
    op2_gemv(mesh, false, 1.0, DGConstants::EMAT, tmp_npf2.dat, 1.0, tmp_grad1.dat);
    op2_gemv(mesh, false, 1.0, DGConstants::EMAT, tmp_npf3.dat, 1.0, tmp_grad2.dat);
    op2_gemv(mesh, false, 1.0, DGConstants::EMAT, tmp_npf0.dat, 0.0, out);
    timer->endTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " mult cells Emat");
  }

  timer->startTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " mult cells");
  timer->startTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " mult cells cells");
  op_par_loop(pmf_3d_mult_cells_geof, "pmf_3d_mult_cells_geof", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_grad0.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(tmp_grad1.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(tmp_grad2.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));

  op2_gemv(mesh, true, 1.0, DGConstants::DR, tmp_grad0.dat, 1.0, out);
  op2_gemv(mesh, true, 1.0, DGConstants::DS, tmp_grad1.dat, 1.0, out);
  op2_gemv(mesh, true, 1.0, DGConstants::DT, tmp_grad2.dat, 1.0, out);
  timer->endTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " mult cells cells");
  timer->endTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " mult cells");
  dg_dat_pool->releaseTempDatCells(tmp_grad0);
  dg_dat_pool->releaseTempDatCells(tmp_grad1);
  dg_dat_pool->releaseTempDatCells(tmp_grad2);
  dg_dat_pool->releaseTempDatCells(tmp_npf0);
  dg_dat_pool->releaseTempDatCells(tmp_npf1);
  dg_dat_pool->releaseTempDatCells(tmp_npf2);
  dg_dat_pool->releaseTempDatCells(tmp_npf3);
  timer->endTimer("FactorPoissonMatrixFreeMultOI3D " + std::to_string(mesh->order_int) + " mult");
}

void FactorPoissonMatrixFreeMultOI3D::mat_free_mult_sp(op_dat in, op_dat out) {
  check_current_order();
  timer->startTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " mult");
  DGTempDat tmp_grad0 = dg_dat_pool->requestTempDatCellsSP(DG_NP);
  DGTempDat tmp_grad1 = dg_dat_pool->requestTempDatCellsSP(DG_NP);
  DGTempDat tmp_grad2 = dg_dat_pool->requestTempDatCellsSP(DG_NP);
  timer->startTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " mult grad");
  op2_gemv_halo_exchange_sp(mesh, false, 1.0, DGConstants::DR, in, 0.0, tmp_grad0.dat);
  op2_gemv_halo_exchange_sp(mesh, false, 1.0, DGConstants::DS, in, 0.0, tmp_grad1.dat);
  op2_gemv_halo_exchange_sp(mesh, false, 1.0, DGConstants::DT, in, 0.0, tmp_grad2.dat);
  if(mesh->order_int == DG_ORDER) {
    // TODO halo compute for this
    DGTempDat tmp_grad0_oi = dg_dat_pool->requestTempDatCellsSP(DG_CUB_3D_NP);
    DGTempDat tmp_grad1_oi = dg_dat_pool->requestTempDatCellsSP(DG_CUB_3D_NP);
    DGTempDat tmp_grad2_oi = dg_dat_pool->requestTempDatCellsSP(DG_CUB_3D_NP);

    op2_gemv_sp(mesh, false, 1.0, DGConstants::CUB3D_INTERP, tmp_grad0.dat, 0.0, tmp_grad0_oi.dat);
    op2_gemv_sp(mesh, false, 1.0, DGConstants::CUB3D_INTERP, tmp_grad1.dat, 0.0, tmp_grad1_oi.dat);
    op2_gemv_sp(mesh, false, 1.0, DGConstants::CUB3D_INTERP, tmp_grad2.dat, 0.0, tmp_grad2_oi.dat);

    op_par_loop(fpmf_3d_grad_oi_sp, "fpmf_3d_grad_oi_sp", mesh->cells,
                op_arg_dat(mesh->geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_factor_oi, -1, OP_ID, DG_CUB_3D_NP, DG_FP_STR, OP_READ),
                op_arg_dat(tmp_grad0_oi.dat, -1, OP_ID, DG_CUB_3D_NP, "float", OP_RW),
                op_arg_dat(tmp_grad1_oi.dat, -1, OP_ID, DG_CUB_3D_NP, "float", OP_RW),
                op_arg_dat(tmp_grad2_oi.dat, -1, OP_ID, DG_CUB_3D_NP, "float", OP_RW));

    op2_gemv_sp(mesh, false, 1.0, DGConstants::CUB3D_PROJ, tmp_grad0_oi.dat, 0.0, tmp_grad0.dat);
    op2_gemv_sp(mesh, false, 1.0, DGConstants::CUB3D_PROJ, tmp_grad1_oi.dat, 0.0, tmp_grad1.dat);
    op2_gemv_sp(mesh, false, 1.0, DGConstants::CUB3D_PROJ, tmp_grad2_oi.dat, 0.0, tmp_grad2.dat);

    dg_dat_pool->releaseTempDatCellsSP(tmp_grad0_oi);
    dg_dat_pool->releaseTempDatCellsSP(tmp_grad1_oi);
    dg_dat_pool->releaseTempDatCellsSP(tmp_grad2_oi);
  } else {
    op_par_loop(fpmf_3d_grad_sp, "fpmf_3d_grad_sp:force_halo_compute", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
              op_arg_dat(mat_free_factor_copy_sp, -1, OP_ID, DG_NP, "float", OP_READ),
              op_arg_dat(tmp_grad0.dat, -1, OP_ID, DG_NP, "float", OP_RW),
              op_arg_dat(tmp_grad1.dat, -1, OP_ID, DG_NP, "float", OP_RW),
              op_arg_dat(tmp_grad2.dat, -1, OP_ID, DG_NP, "float", OP_RW));
  }
  timer->endTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " mult grad");

  DGTempDat tmp_npf0 = dg_dat_pool->requestTempDatCellsSP(DG_NUM_FACES * DG_NPF);
  DGTempDat tmp_npf1 = dg_dat_pool->requestTempDatCellsSP(DG_NUM_FACES * DG_NPF);
  DGTempDat tmp_npf2 = dg_dat_pool->requestTempDatCellsSP(DG_NUM_FACES * DG_NPF);
  DGTempDat tmp_npf3 = dg_dat_pool->requestTempDatCellsSP(DG_NUM_FACES * DG_NPF);

  op_par_loop(zero_npf_1_sp, "zero_npf_sp", mesh->cells,
              op_arg_dat(tmp_npf0.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_WRITE));

  op_par_loop(zero_npf_3_sp, "zero_npf_3_sp", mesh->cells,
              op_arg_dat(tmp_npf1.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_WRITE),
              op_arg_dat(tmp_npf2.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_WRITE),
              op_arg_dat(tmp_npf3.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_WRITE));

  timer->startTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " mult faces");
  mesh->jump_sp(in, tmp_npf0.dat);
  mesh->avg_sp(tmp_grad0.dat, tmp_npf1.dat);
  mesh->avg_sp(tmp_grad1.dat, tmp_npf2.dat);
  mesh->avg_sp(tmp_grad2.dat, tmp_npf3.dat);
  timer->endTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " mult faces");

  timer->startTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " mult bfaces");
  if(mesh->bface2cells) {
    op_par_loop(pmf_3d_mult_avg_jump_sp, "pmf_3d_mult_avg_jump_sp", mesh->bfaces,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_dat(mat_free_bcs, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(mesh->bfaceNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(in, 0, mesh->bface2cells, DG_NP, "float", OP_READ),
                op_arg_dat(tmp_grad0.dat, 0, mesh->bface2cells, DG_NP, "float", OP_READ),
                op_arg_dat(tmp_grad1.dat, 0, mesh->bface2cells, DG_NP, "float", OP_READ),
                op_arg_dat(tmp_grad2.dat, 0, mesh->bface2cells, DG_NP, "float", OP_READ),
                op_arg_dat(tmp_npf0.dat, 0, mesh->bface2cells, DG_NUM_FACES * DG_NPF, "float", OP_INC),
                op_arg_dat(tmp_npf1.dat, 0, mesh->bface2cells, DG_NUM_FACES * DG_NPF, "float", OP_INC),
                op_arg_dat(tmp_npf2.dat, 0, mesh->bface2cells, DG_NUM_FACES * DG_NPF, "float", OP_INC),
                op_arg_dat(tmp_npf3.dat, 0, mesh->bface2cells, DG_NUM_FACES * DG_NPF, "float", OP_INC));
  }
  timer->endTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " mult bfaces");

  if(mesh->order_int == DG_ORDER) {
    DGTempDat jump_cub = dg_dat_pool->requestTempDatCellsSP(DG_NUM_FACES * DG_CUB_SURF_3D_NP);
    DGTempDat fluxX_cub = dg_dat_pool->requestTempDatCellsSP(DG_NUM_FACES * DG_CUB_SURF_3D_NP);
    DGTempDat fluxY_cub = dg_dat_pool->requestTempDatCellsSP(DG_NUM_FACES * DG_CUB_SURF_3D_NP);
    DGTempDat fluxZ_cub = dg_dat_pool->requestTempDatCellsSP(DG_NUM_FACES * DG_CUB_SURF_3D_NP);
    op2_gemv_sp(mesh, false, 1.0, DGConstants::CUBSURF3D_INTERP, tmp_npf0.dat, 0.0, jump_cub.dat);
    op_par_loop(fpmf_3d_mult_flux_oi_sp_0, "fpmf_3d_mult_flux_oi_sp_0", mesh->cells,
                op_arg_dat(mesh->nx_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->ny_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->nz_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->sJ_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_factor_surf_oi, -1, OP_ID, DG_NUM_FACES * DG_CUB_SURF_3D_NP, DG_FP_STR, OP_READ),
                op_arg_dat(jump_cub.dat,  -1, OP_ID, DG_NUM_FACES * DG_CUB_SURF_3D_NP, "float", OP_READ),
                op_arg_dat(fluxX_cub.dat, -1, OP_ID, DG_NUM_FACES * DG_CUB_SURF_3D_NP, "float", OP_WRITE),
                op_arg_dat(fluxY_cub.dat, -1, OP_ID, DG_NUM_FACES * DG_CUB_SURF_3D_NP, "float", OP_WRITE),
                op_arg_dat(fluxZ_cub.dat, -1, OP_ID, DG_NUM_FACES * DG_CUB_SURF_3D_NP, "float", OP_WRITE));
    op2_gemv_sp(mesh, false, 1.0, DGConstants::CUBSURF3D_LIFT, fluxX_cub.dat, 1.0, tmp_grad0.dat);
    op2_gemv_sp(mesh, false, 1.0, DGConstants::CUBSURF3D_LIFT, fluxY_cub.dat, 1.0, tmp_grad1.dat);
    op2_gemv_sp(mesh, false, 1.0, DGConstants::CUBSURF3D_LIFT, fluxZ_cub.dat, 1.0, tmp_grad2.dat);
    dg_dat_pool->releaseTempDatCellsSP(jump_cub);
    dg_dat_pool->releaseTempDatCellsSP(fluxX_cub);
    dg_dat_pool->releaseTempDatCellsSP(fluxY_cub);
    dg_dat_pool->releaseTempDatCellsSP(fluxZ_cub);
    op_par_loop(fpmf_3d_mult_flux_oi_sp_1, "fpmf_3d_mult_flux_oi_sp_1", mesh->cells,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_dat(mesh->nx_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->ny_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->nz_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->sJ_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_tau_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(tmp_npf0.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_RW),
                op_arg_dat(tmp_npf1.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_RW),
                op_arg_dat(tmp_npf2.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_RW),
                op_arg_dat(tmp_npf3.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_RW));
    mesh->mass_sp(tmp_grad0.dat);
    mesh->mass_sp(tmp_grad1.dat);
    mesh->mass_sp(tmp_grad2.dat);
    op2_gemv_sp(mesh, false, 1.0, DGConstants::EMAT, tmp_npf0.dat, 0.0, out);
  } else {
    timer->startTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " finish flux");
    op_par_loop(fpmf_3d_mult_flux_sp, "fpmf_3d_mult_flux_sp", mesh->cells,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_dat(mesh->nx_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->ny_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->nz_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->sJ_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_tau_c, -1, OP_ID, 4, DG_FP_STR, OP_READ),
                op_arg_dat(mat_free_factor_copy_sp, -1, OP_ID, DG_NP, "float", OP_READ),
                op_arg_dat(tmp_npf0.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_RW),
                op_arg_dat(tmp_npf1.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_RW),
                op_arg_dat(tmp_npf2.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_RW),
                op_arg_dat(tmp_npf3.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, "float", OP_RW));
    timer->endTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " finish flux");

    timer->startTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " mult cells MM");
    mesh->mass_sp(tmp_grad0.dat);
    mesh->mass_sp(tmp_grad1.dat);
    mesh->mass_sp(tmp_grad2.dat);
    timer->endTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " mult cells MM");

    timer->startTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " mult cells Emat");
    op2_gemv_sp(mesh, false, 1.0, DGConstants::EMAT, tmp_npf1.dat, 1.0, tmp_grad0.dat);
    op2_gemv_sp(mesh, false, 1.0, DGConstants::EMAT, tmp_npf2.dat, 1.0, tmp_grad1.dat);
    op2_gemv_sp(mesh, false, 1.0, DGConstants::EMAT, tmp_npf3.dat, 1.0, tmp_grad2.dat);
    op2_gemv_sp(mesh, false, 1.0, DGConstants::EMAT, tmp_npf0.dat, 0.0, out);
    timer->endTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " mult cells Emat");
  }

  timer->startTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " mult cells");
  timer->startTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " mult cells cells");
  op_par_loop(pmf_3d_mult_cells_geof_sp, "pmf_3d_mult_cells_geof_sp", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(mesh->geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_grad0.dat, -1, OP_ID, DG_NP, "float", OP_RW),
              op_arg_dat(tmp_grad1.dat, -1, OP_ID, DG_NP, "float", OP_RW),
              op_arg_dat(tmp_grad2.dat, -1, OP_ID, DG_NP, "float", OP_RW));

  op2_gemv_sp(mesh, true, 1.0, DGConstants::DR, tmp_grad0.dat, 1.0, out);
  op2_gemv_sp(mesh, true, 1.0, DGConstants::DS, tmp_grad1.dat, 1.0, out);
  op2_gemv_sp(mesh, true, 1.0, DGConstants::DT, tmp_grad2.dat, 1.0, out);
  timer->endTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " mult cells cells");
  timer->endTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " mult cells");
  dg_dat_pool->releaseTempDatCellsSP(tmp_grad0);
  dg_dat_pool->releaseTempDatCellsSP(tmp_grad1);
  dg_dat_pool->releaseTempDatCellsSP(tmp_grad2);
  dg_dat_pool->releaseTempDatCellsSP(tmp_npf0);
  dg_dat_pool->releaseTempDatCellsSP(tmp_npf1);
  dg_dat_pool->releaseTempDatCellsSP(tmp_npf2);
  dg_dat_pool->releaseTempDatCellsSP(tmp_npf3);
  timer->endTimer("FactorPoissonMatrixFreeMultOI3D sp" + std::to_string(mesh->order_int) + " mult");
}
