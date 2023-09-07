#include "dg_mesh/dg_mesh_2d.h"

#include "op_seq.h"

#include "dg_compiler_defs.h"
#include "dg_op2_blas.h"
#include "dg_constants/dg_constants.h"
#include "dg_dat_pool.h"

extern DGConstants *constants;
extern DGDatPool *dg_dat_pool;

void DGMesh2D::div(op_dat u, op_dat v, op_dat res) {
  DGTempDat tmp_r = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_s = dg_dat_pool->requestTempDatCells(DG_NP);

  op_par_loop(div_2d_geof, "div_2d_geof", cells,
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(geof, -1, OP_ID, 5, DG_FP_STR, OP_READ),
              op_arg_dat(u, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(v, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_r.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(tmp_s.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::DR, tmp_r.dat, 0.0, res);
  op2_gemv(this, false, 1.0, DGConstants::DS, tmp_s.dat, 1.0, res);

  dg_dat_pool->releaseTempDatCells(tmp_r);
  dg_dat_pool->releaseTempDatCells(tmp_s);
/*
  op_par_loop(div_2d, "div_2d", cells,
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::DR), DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::DS), DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(u,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(v,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rx, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(sx, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ry, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(sy, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(res, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));
*/
}

void DGMesh2D::div_weak(op_dat u, op_dat v, op_dat res) {
  DGTempDat tmp_r = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_s = dg_dat_pool->requestTempDatCells(DG_NP);

  op_par_loop(div_2d_geof, "div_2d_geof", cells,
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(geof, -1, OP_ID, 5, DG_FP_STR, OP_READ),
              op_arg_dat(u, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(v, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_r.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(tmp_s.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::DRW, tmp_r.dat, 0.0, res);
  op2_gemv(this, false, 1.0, DGConstants::DSW, tmp_s.dat, 1.0, res);

  dg_dat_pool->releaseTempDatCells(tmp_r);
  dg_dat_pool->releaseTempDatCells(tmp_s);
}

void DGMesh2D::div_with_central_flux(op_dat u, op_dat v, op_dat res) {
  div(u, v, res);

  // Central flux
  DGTempDat tmp_0 = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);

  op_par_loop(zero_npf, "zero_npf", cells,
              op_arg_dat(tmp_0.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));

  op_par_loop(div_2d_central_flux, "div_2d_central_flux", faces,
              op_arg_dat(edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(reverse, -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(nx, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(ny, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(fscale, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(u, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(v, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_0.dat, -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, -1.0, DGConstants::LIFT, tmp_0.dat, 1.0, res);

  dg_dat_pool->releaseTempDatCells(tmp_0);
}

void DGMesh2D::curl(op_dat u, op_dat v, op_dat res) {
  // Same matrix multiplications as div
  op2_gemv(this, false, 1.0, DGConstants::DR, u, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DS, u, 0.0, op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::DR, v, 0.0, op_tmp[2]);
  op2_gemv(this, false, 1.0, DGConstants::DS, v, 0.0, op_tmp[3]);

  op_par_loop(curl_2d, "curl_2d", cells,
              op_arg_dat(order,     -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[2], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[3], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rx,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(sx,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ry,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(sy,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(res, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));
}

void DGMesh2D::grad(op_dat u, op_dat ux, op_dat uy) {
  DGTempDat tmp_r = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_s = dg_dat_pool->requestTempDatCells(DG_NP);

  op2_gemv(this, false, 1.0, DGConstants::DR, u, 0.0, tmp_r.dat);
  op2_gemv(this, false, 1.0, DGConstants::DS, u, 0.0, tmp_s.dat);

  op_par_loop(grad_2d_geof, "grad_2d_geof", cells,
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(geof, -1, OP_ID, 5, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_r.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_s.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ux, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(uy, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  dg_dat_pool->releaseTempDatCells(tmp_r);
  dg_dat_pool->releaseTempDatCells(tmp_s);
/*
  op_par_loop(grad_2d, "grad_2d", cells,
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::DR), DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::DS), DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(u,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rx, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(sx, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ry, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(sy, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ux, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(uy, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));
*/
}

void DGMesh2D::grad_with_central_flux(op_dat u, op_dat ux, op_dat uy) {
  grad(u, ux, uy);

  // Central flux
  DGTempDat tmp_0 = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);
  DGTempDat tmp_1 = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);

  op_par_loop(zero_npf, "zero_npf", cells,
              op_arg_dat(tmp_0.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));
  op_par_loop(zero_npf, "zero_npf", cells,
              op_arg_dat(tmp_1.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));

  op_par_loop(grad_2d_central_flux, "grad_2d_central_flux", faces,
              op_arg_dat(edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(reverse, -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(nx, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(ny, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(fscale, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(u, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_0.dat, -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(tmp_1.dat, -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, -1.0, DGConstants::LIFT, tmp_0.dat, 1.0, ux);
  op2_gemv(this, false, -1.0, DGConstants::LIFT, tmp_1.dat, 1.0, uy);

  dg_dat_pool->releaseTempDatCells(tmp_0);
  dg_dat_pool->releaseTempDatCells(tmp_1);
}

void DGMesh2D::grad_over_int_with_central_flux(op_dat u, op_dat ux, op_dat uy) {
  DGTempDat vol_u = dg_dat_pool->requestTempDatCells(DG_CUB_2D_NP);
  op2_gemv(this, false, 1.0, DGConstants::CUB2D_INTERP, u, 0.0, vol_u.dat);
  
  op2_gemv(this, false, 1.0, DGConstants::CUB2D_PDR, vol_u.dat, 0.0, ux);
  op2_gemv(this, false, 1.0, DGConstants::CUB2D_PDS, vol_u.dat, 0.0, uy);

  dg_dat_pool->releaseTempDatCells(vol_u);

  op_par_loop(grad_over_int_2d_0, "grad_over_int_2d_0", cells,
              op_arg_dat(geof, -1, OP_ID, 5, DG_FP_STR, OP_READ),
              op_arg_dat(ux,   -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(uy,   -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));

  DGTempDat uM = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);
  DGTempDat uP = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);

  op_par_loop(grad_over_int_2d_1, "grad_over_int_2d_1", faces,
              op_arg_dat(edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(reverse, -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(u, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(uM.dat, -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(uP.dat, -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));
  
  if(bface2cells) {
    op_par_loop(grad_over_int_2d_2, "grad_over_int_2d_2", bfaces,
                op_arg_dat(bedgeNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(u, 0, bface2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(uM.dat, 0, bface2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_RW),
                op_arg_dat(uP.dat, 0, bface2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_RW));
  }

  DGTempDat uM_cub = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_CUB_SURF_2D_NP);
  DGTempDat uP_cub = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_CUB_SURF_2D_NP);

  op2_gemv(this, false, 1.0, DGConstants::CUBSURF2D_INTERP, uM.dat, 0.0, uM_cub.dat);
  op2_gemv(this, false, 1.0, DGConstants::CUBSURF2D_INTERP, uP.dat, 0.0, uP_cub.dat);

  dg_dat_pool->releaseTempDatCells(uM);
  dg_dat_pool->releaseTempDatCells(uP);

  op_par_loop(grad_over_int_2d_3, "grad_over_int_2d_3", cells,
              op_arg_dat(nx_c_new, -1, OP_ID, 3, DG_FP_STR, OP_READ),
              op_arg_dat(ny_c_new, -1, OP_ID, 3, DG_FP_STR, OP_READ),
              op_arg_dat(sJ_c_new, -1, OP_ID, 3, DG_FP_STR, OP_READ),
              op_arg_dat(geof, -1, OP_ID, 5, DG_FP_STR, OP_READ),
              op_arg_dat(uM_cub.dat, -1, OP_ID, DG_NUM_FACES * DG_CUB_SURF_2D_NP, DG_FP_STR, OP_RW),
              op_arg_dat(uP_cub.dat, -1, OP_ID, DG_NUM_FACES * DG_CUB_SURF_2D_NP, DG_FP_STR, OP_RW));

  op2_gemv(this, false, 1.0, DGConstants::CUBSURF2D_LIFT, uM_cub.dat, -1.0, ux);
  op2_gemv(this, false, 1.0, DGConstants::CUBSURF2D_LIFT, uP_cub.dat, -1.0, uy);

  dg_dat_pool->releaseTempDatCells(uM_cub);
  dg_dat_pool->releaseTempDatCells(uP_cub);
}

void DGMesh2D::mass(op_dat u) {
  op_par_loop(J, "J", cells,
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(J, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(u, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::MASS, op_tmp[0], 0.0, u);
}

void DGMesh2D::mass_sp(op_dat u) {
  DGTempDat tmp_dat = dg_dat_pool->requestTempDatCellsSP(DG_NP);
  op_par_loop(J_sp, "J_sp", cells,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(geof, -1, OP_ID, 5, DG_FP_STR, OP_READ),
              op_arg_dat(u, -1, OP_ID, DG_NP, "float", OP_READ),
              op_arg_dat(tmp_dat.dat, -1, OP_ID, DG_NP, "float", OP_WRITE));

  op2_gemv_sp(this, false, 1.0, DGConstants::MASS, tmp_dat.dat, 0.0, u);
  dg_dat_pool->releaseTempDatCellsSP(tmp_dat);
}

void DGMesh2D::inv_mass(op_dat u) {
  op_par_loop(inv_J, "inv_J", cells,
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(J, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(u, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::INV_MASS, op_tmp[0], 0.0, u);
}

void DGMesh2D::avg(op_dat in, op_dat out) {
  op_par_loop(avg_2d, "avg_2d", faces,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(reverse, -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(in,  -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(out, -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));
}

void DGMesh2D::jump(op_dat in, op_dat out) {
  op_par_loop(jump_2d, "jump_2d", faces,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(reverse, -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(in,  -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(out, -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));
}

void DGMesh2D::avg_sp(op_dat in, op_dat out) {
  op_par_loop(avg_2d_sp, "avg_2d_sp", faces,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(reverse, -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(in,  -2, face2cells, DG_NP, "float", OP_READ),
              op_arg_dat(out, -2, face2cells, DG_NUM_FACES * DG_NPF, "float", OP_WRITE));
}

void DGMesh2D::jump_sp(op_dat in, op_dat out) {
  op_par_loop(jump_2d_sp, "jump_2d_sp", faces,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(reverse, -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(in,  -2, face2cells, DG_NP, "float", OP_READ),
              op_arg_dat(out, -2, face2cells, DG_NUM_FACES * DG_NPF, "float", OP_WRITE));
}

void DGMesh2D::interp_dat_between_orders(int old_order, int new_order, op_dat in, op_dat out) {
  op2_gemv_interp(this, old_order, new_order, in, out);
}

void DGMesh2D::interp_dat_between_orders(int old_order, int new_order, op_dat in) {
  DGTempDat tmp_0 = dg_dat_pool->requestTempDatCells(DG_NP);
  op2_gemv_interp(this, old_order, new_order, in, tmp_0.dat);

  op_par_loop(copy_dg_np_tk, "copy_dg_np_tk", cells,
              op_arg_dat(tmp_0.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(in, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  dg_dat_pool->releaseTempDatCells(tmp_0);
}

void DGMesh2D::interp_dat_between_orders_sp(int old_order, int new_order, op_dat in) {
  DGTempDat tmp_0 = dg_dat_pool->requestTempDatCellsSP(DG_NP);
  op2_gemv_interp_sp(this, old_order, new_order, in, tmp_0.dat);

  op_par_loop(copy_dg_np_sp_tk, "copy_dg_np_tk", cells,
              op_arg_dat(tmp_0.dat, -1, OP_ID, DG_NP, "float", OP_READ),
              op_arg_dat(in, -1, OP_ID, DG_NP, "float", OP_WRITE));

  dg_dat_pool->releaseTempDatCellsSP(tmp_0);
}