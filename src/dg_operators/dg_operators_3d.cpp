#include "dg_mesh/dg_mesh_3d.h"

#include "op_seq.h"

#include "dg_compiler_defs.h"
#include "dg_op2_blas.h"
#include "dg_constants/dg_constants.h"
#include "dg_dat_pool.h"

extern DGConstants *constants;
extern DGDatPool *dg_dat_pool;

void custom_kernel_grad_3d(const int order, char const *name, op_set set,
  op_arg arg0,
  op_arg arg4,
  op_arg arg5,
  op_arg arg6,
  op_arg arg7,
  op_arg arg8,
  op_arg arg9,
  op_arg arg10,
  op_arg arg11,
  op_arg arg12,
  op_arg arg13,
  op_arg arg14,
  op_arg arg15,
  op_arg arg16);

void custom_kernel_mass(const int order, char const *name, op_set set,
  op_arg arg0,
  op_arg arg2,
  op_arg arg3);

void DGMesh3D::grad(op_dat u, op_dat ux, op_dat uy, op_dat uz) {
#if defined(OP2_DG_CUDA) && !defined(DG_OP2_SOA)
custom_kernel_grad_3d(order_int, "grad_3d",cells,
                     op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ),
                     op_arg_dat(u,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
                     op_arg_dat(rx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                     op_arg_dat(sx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                     op_arg_dat(tx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                     op_arg_dat(ry, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                     op_arg_dat(sy, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                     op_arg_dat(ty, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                     op_arg_dat(rz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                     op_arg_dat(sz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                     op_arg_dat(tz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                     op_arg_dat(ux, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
                     op_arg_dat(uy, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
                     op_arg_dat(uz, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));
#else
  DGTempDat tmp_r = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_s = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_t = dg_dat_pool->requestTempDatCells(DG_NP);

  op2_gemv(this, false, 1.0, DGConstants::DR, u, 0.0, tmp_r.dat);
  op2_gemv(this, false, 1.0, DGConstants::DS, u, 0.0, tmp_s.dat);
  op2_gemv(this, false, 1.0, DGConstants::DT, u, 0.0, tmp_t.dat);
  op_par_loop(grad_3d_geof, "grad_3d_geof", cells,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_r.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_s.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_t.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ux, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(uy, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(uz, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  dg_dat_pool->releaseTempDatCells(tmp_r);
  dg_dat_pool->releaseTempDatCells(tmp_s);
  dg_dat_pool->releaseTempDatCells(tmp_t);
  /*
  op_par_loop(grad_3d, "grad_3d", cells,
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::DR), DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::DS), DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::DT), DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(u,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(ry, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sy, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(ty, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(rz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(ux, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(uy, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(uz, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));
  */
#endif
}

void DGMesh3D::grad_weak(op_dat u, op_dat ux, op_dat uy, op_dat uz) {
  DGTempDat tmp_r = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_s = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_t = dg_dat_pool->requestTempDatCells(DG_NP);

  op2_gemv(this, false, 1.0, DGConstants::DRW, u, 0.0, tmp_r.dat);
  op2_gemv(this, false, 1.0, DGConstants::DSW, u, 0.0, tmp_s.dat);
  op2_gemv(this, false, 1.0, DGConstants::DTW, u, 0.0, tmp_t.dat);
  op_par_loop(grad_3d_geof, "grad_3d_geof", cells,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_r.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_s.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_t.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ux, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(uy, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(uz, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  dg_dat_pool->releaseTempDatCells(tmp_r);
  dg_dat_pool->releaseTempDatCells(tmp_s);
  dg_dat_pool->releaseTempDatCells(tmp_t);
}

void DGMesh3D::grad_with_central_flux(op_dat u, op_dat ux, op_dat uy, op_dat uz) {
  grad(u, ux, uy, uz);

  DGTempDat tmp_0 = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);
  DGTempDat tmp_1 = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);
  DGTempDat tmp_2 = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);

  op_par_loop(zero_npf, "zero_npf", cells,
              op_arg_dat(tmp_0.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));
  op_par_loop(zero_npf, "zero_npf", cells,
              op_arg_dat(tmp_1.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));
  op_par_loop(zero_npf, "zero_npf", cells,
              op_arg_dat(tmp_2.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));

  op_par_loop(grad_3d_central_flux, "grad_3d_central_flux", faces,
              op_arg_dat(faceNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(fmaskL,  -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(fmaskR,  -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(nx, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(ny, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(nz, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(fscale, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(u, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_0.dat, -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC),
              op_arg_dat(tmp_1.dat, -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC),
              op_arg_dat(tmp_2.dat, -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC));

  op2_gemv(this, false, -1.0, DGConstants::LIFT, tmp_0.dat, 1.0, ux);
  op2_gemv(this, false, -1.0, DGConstants::LIFT, tmp_1.dat, 1.0, uy);
  op2_gemv(this, false, -1.0, DGConstants::LIFT, tmp_2.dat, 1.0, uz);

  dg_dat_pool->releaseTempDatCells(tmp_0);
  dg_dat_pool->releaseTempDatCells(tmp_1);
  dg_dat_pool->releaseTempDatCells(tmp_2);
}

void DGMesh3D::div(op_dat u, op_dat v, op_dat w, op_dat res) {
  DGTempDat tmp_r = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_s = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_t = dg_dat_pool->requestTempDatCells(DG_NP);

  op_par_loop(div_3d, "div_3d", cells,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(u,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(v,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(w,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_r.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(tmp_s.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(tmp_t.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::DR, tmp_r.dat, 0.0, res);
  op2_gemv(this, false, 1.0, DGConstants::DS, tmp_s.dat, 1.0, res);
  op2_gemv(this, false, 1.0, DGConstants::DT, tmp_t.dat, 1.0, res);

  dg_dat_pool->releaseTempDatCells(tmp_r);
  dg_dat_pool->releaseTempDatCells(tmp_s);
  dg_dat_pool->releaseTempDatCells(tmp_t);
}

void DGMesh3D::div_with_central_flux(op_dat u, op_dat v, op_dat w, op_dat res) {
  div(u, v, w, res);

  DGTempDat tmp_0 = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);

  op_par_loop(zero_npf, "zero_npf", cells,
              op_arg_dat(tmp_0.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));

  op_par_loop(div_3d_central_flux, "div_3d_central_flux", faces,
              op_arg_dat(faceNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(fmaskL,  -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(fmaskR,  -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(nx, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(ny, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(nz, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(fscale, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(u, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(v, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(w, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_0.dat, -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC));

  op2_gemv(this, false, -1.0, DGConstants::LIFT, tmp_0.dat, 1.0, res);

  dg_dat_pool->releaseTempDatCells(tmp_0);
}

void DGMesh3D::div_weak(op_dat u, op_dat v, op_dat w, op_dat res) {
  DGTempDat tmp_r = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_s = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_t = dg_dat_pool->requestTempDatCells(DG_NP);

  op_par_loop(div_3d, "div_3d", cells,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(u,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(v,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(w,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_r.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(tmp_s.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(tmp_t.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::DRW, tmp_r.dat, 0.0, res);
  op2_gemv(this, false, 1.0, DGConstants::DSW, tmp_s.dat, 1.0, res);
  op2_gemv(this, false, 1.0, DGConstants::DTW, tmp_t.dat, 1.0, res);

  dg_dat_pool->releaseTempDatCells(tmp_r);
  dg_dat_pool->releaseTempDatCells(tmp_s);
  dg_dat_pool->releaseTempDatCells(tmp_t);
}

void DGMesh3D::div_weak_with_central_flux(op_dat u, op_dat v, op_dat w, op_dat res) {
  div_weak(u, v, w, res);

  DGTempDat tmp_0 = dg_dat_pool->requestTempDatCells(DG_NUM_FACES * DG_NPF);

  op_par_loop(zero_npf, "zero_npf", cells,
              op_arg_dat(tmp_0.dat, -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));

  op_par_loop(div_weak_3d_central_flux, "div_weak_3d_central_flux", faces,
              op_arg_dat(faceNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(fmaskL,  -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(fmaskR,  -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(nx, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(ny, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(nz, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(fscale, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(u, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(v, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(w, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_0.dat, -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC));

  // TODO bfaces
  op2_gemv(this, false, 1.0, DGConstants::LIFT, tmp_0.dat, -1.0, res);

  dg_dat_pool->releaseTempDatCells(tmp_0);
}

void DGMesh3D::curl(op_dat u, op_dat v, op_dat w,
          op_dat resx, op_dat resy, op_dat resz) {
  DGTempDat tmp_r = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_s = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_t = dg_dat_pool->requestTempDatCells(DG_NP);

  op2_gemv(this, false, 1.0, DGConstants::DR, u, 0.0, tmp_r.dat);
  op2_gemv(this, false, 1.0, DGConstants::DS, u, 0.0, tmp_s.dat);
  op2_gemv(this, false, 1.0, DGConstants::DT, u, 0.0, tmp_t.dat);

  op_par_loop(curl0_3d, "curl0_3d", cells,
              op_arg_dat(tmp_r.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_s.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_t.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ry, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sy, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(ty, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(rz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(resy, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(resz, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::DR, v, 0.0, tmp_r.dat);
  op2_gemv(this, false, 1.0, DGConstants::DS, v, 0.0, tmp_s.dat);
  op2_gemv(this, false, 1.0, DGConstants::DT, v, 0.0, tmp_t.dat);

  op_par_loop(curl1_3d, "curl1_3d", cells,
              op_arg_dat(tmp_r.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_s.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_t.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(rz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(resx, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(resz, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));

  op2_gemv(this, false, 1.0, DGConstants::DR, w, 0.0, tmp_r.dat);
  op2_gemv(this, false, 1.0, DGConstants::DS, w, 0.0, tmp_s.dat);
  op2_gemv(this, false, 1.0, DGConstants::DT, w, 0.0, tmp_t.dat);

  op_par_loop(curl2_3d, "curl2_3d", cells,
              op_arg_dat(tmp_r.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_s.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_t.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(ry, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sy, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(ty, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(resx, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(resy, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));

  dg_dat_pool->releaseTempDatCells(tmp_r);
  dg_dat_pool->releaseTempDatCells(tmp_s);
  dg_dat_pool->releaseTempDatCells(tmp_t);
}

void DGMesh3D::mass(op_dat u) {
  #if defined(OP2_DG_CUDA) && !defined(DG_OP2_SOA)
  custom_kernel_mass(order_int, "mass", cells,
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(J, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(u, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));

  #else
  DGTempDat tmp_0 = dg_dat_pool->requestTempDatCells(DG_NP);

  op2_gemv(this, false, 1.0, DGConstants::MASS, u, 0.0, tmp_0.dat);
  op_par_loop(J_3d, "J_3d", cells,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(J, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_0.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(u, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  dg_dat_pool->releaseTempDatCells(tmp_0);
  /*
  op_par_loop(mass, "mass", cells,
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::MASS), DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(J, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(u, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  */
  #endif
}

void DGMesh3D::mass_sp(op_dat u) {
  DGTempDat tmp_0 = dg_dat_pool->requestTempDatCellsSP(DG_NP);

  op2_gemv_sp(this, false, 1.0, DGConstants::MASS, u, 0.0, tmp_0.dat);
  op_par_loop(J_3d_sp, "J_3d_sp", cells,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(J, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_0.dat, -1, OP_ID, DG_NP, "float", OP_READ),
              op_arg_dat(u, -1, OP_ID, DG_NP, "float", OP_WRITE));

  dg_dat_pool->releaseTempDatCellsSP(tmp_0);
}

void DGMesh3D::inv_mass(op_dat u) {
  op_par_loop(inv_mass, "inv_mass", cells,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(J, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(u, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
}

void DGMesh3D::interp_dat_between_orders(int old_order, int new_order, op_dat in, op_dat out) {
  op2_gemv_interp(this, old_order, new_order, in, out);
}

void DGMesh3D::interp_dat_between_orders(int old_order, int new_order, op_dat in) {
  DGTempDat tmp_0 = dg_dat_pool->requestTempDatCells(DG_NP);
  op2_gemv_interp(this, old_order, new_order, in, tmp_0.dat);

  op_par_loop(copy_dg_np_tk, "copy_dg_np_tk", cells,
              op_arg_dat(tmp_0.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(in, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  dg_dat_pool->releaseTempDatCells(tmp_0);
}

void DGMesh3D::interp_dat_between_orders_sp(int old_order, int new_order, op_dat in) {
  DGTempDat tmp_0 = dg_dat_pool->requestTempDatCellsSP(DG_NP);
  op2_gemv_interp_sp(this, old_order, new_order, in, tmp_0.dat);

  op_par_loop(copy_dg_np_sp_tk, "copy_dg_np_tk", cells,
              op_arg_dat(tmp_0.dat, -1, OP_ID, DG_NP, "float", OP_READ),
              op_arg_dat(in, -1, OP_ID, DG_NP, "float", OP_WRITE));

  dg_dat_pool->releaseTempDatCellsSP(tmp_0);
}

void DGMesh3D::avg(op_dat in, op_dat out) {
  op_par_loop(avg_3d, "avg_3d", faces,
              op_arg_dat(order, -2, face2cells, 1, "int", OP_READ),
              op_arg_dat(faceNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(fmaskL, -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(fmaskR, -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(in,  -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(out, -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));
}

void DGMesh3D::jump(op_dat in, op_dat out) {
  op_par_loop(jump_3d, "jump_3d", faces,
              op_arg_dat(order, -2, face2cells, 1, "int", OP_READ),
              op_arg_dat(faceNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(fmaskL, -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(fmaskR, -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(in,  -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(out, -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));
}

void DGMesh3D::avg_sp(op_dat in, op_dat out) {
  op_par_loop(avg_3d_sp, "avg_3d", faces,
              op_arg_dat(order, -2, face2cells, 1, "int", OP_READ),
              op_arg_dat(faceNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(fmaskL, -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(fmaskR, -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(in,  -2, face2cells, DG_NP, "float", OP_READ),
              op_arg_dat(out, -2, face2cells, DG_NUM_FACES * DG_NPF, "float", OP_WRITE));
}

void DGMesh3D::jump_sp(op_dat in, op_dat out) {
  op_par_loop(jump_3d_sp, "jump_3d", faces,
              op_arg_dat(order, -2, face2cells, 1, "int", OP_READ),
              op_arg_dat(faceNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(fmaskL, -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(fmaskR, -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(in,  -2, face2cells, DG_NP, "float", OP_READ),
              op_arg_dat(out, -2, face2cells, DG_NUM_FACES * DG_NPF, "float", OP_WRITE));
}

void DGMesh3D::grad_halo_exchange(op_dat u, op_dat ux, op_dat uy, op_dat uz) {
  DGTempDat tmp_r = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_s = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_t = dg_dat_pool->requestTempDatCells(DG_NP);

  op2_gemv_halo_exchange(this, false, 1.0, DGConstants::DR, u, 0.0, tmp_r.dat);
  op2_gemv_halo_exchange(this, false, 1.0, DGConstants::DS, u, 0.0, tmp_s.dat);
  op2_gemv_halo_exchange(this, false, 1.0, DGConstants::DT, u, 0.0, tmp_t.dat);

  op_par_loop(grad_3d_he, "grad_3d_he:force_halo_compute", cells,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_r.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_s.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_t.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ux, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(uy, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(uz, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  dg_dat_pool->releaseTempDatCells(tmp_r);
  dg_dat_pool->releaseTempDatCells(tmp_s);
  dg_dat_pool->releaseTempDatCells(tmp_t);
}
