#include "dg_mesh/dg_mesh_3d.h"

#include "op_seq.h"

#include "dg_compiler_defs.h"
#include "dg_op2_blas.h"
#include "dg_constants/dg_constants.h"

extern DGConstants *constants;

void DGMesh3D::grad(op_dat u, op_dat ux, op_dat uy, op_dat uz) {
  // op2_gemv(this, false, 1.0, DGConstants::DR, u, 0.0, op_tmp[0]);
  // op2_gemv(this, false, 1.0, DGConstants::DS, u, 0.0, op_tmp[1]);
  // op2_gemv(this, false, 1.0, DGConstants::DT, u, 0.0, op_tmp[2]);

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
}

void DGMesh3D::grad_with_central_flux(op_dat u, op_dat ux, op_dat uy, op_dat uz) {
  grad(u, ux, uy, uz);

  op_par_loop(zero_npf, "zero_npf", cells,
              op_arg_dat(op_tmp_npf[0], -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));
  op_par_loop(zero_npf, "zero_npf", cells,
              op_arg_dat(op_tmp_npf[1], -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));
  op_par_loop(zero_npf, "zero_npf", cells,
              op_arg_dat(op_tmp_npf[2], -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));

  op_par_loop(grad_3d_central_flux, "grad_3d_central_flux", faces,
              op_arg_dat(faceNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(fmaskL,  -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(fmaskR,  -1, OP_ID, DG_NPF, "int", OP_READ),
              op_arg_dat(nx, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(ny, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(nz, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(fscale, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(u, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp_npf[0], -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC),
              op_arg_dat(op_tmp_npf[1], -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC),
              op_arg_dat(op_tmp_npf[2], -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC));

  op2_gemv(this, false, 1.0, DGConstants::LIFT, op_tmp_npf[0], 1.0, ux);
  op2_gemv(this, false, 1.0, DGConstants::LIFT, op_tmp_npf[1], 1.0, uy);
  op2_gemv(this, false, 1.0, DGConstants::LIFT, op_tmp_npf[2], 1.0, uz);
}

void DGMesh3D::div(op_dat u, op_dat v, op_dat w, op_dat res) {
  op2_gemv(this, false, 1.0, DGConstants::DR, u, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DS, u, 0.0, op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::DT, u, 0.0, op_tmp[2]);

  op_par_loop(zero_np, "zero_np", cells,
              op_arg_dat(res, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  op_par_loop(div_3d, "div_3d", cells,
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[2], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(res, -1, OP_ID, DG_NP, DG_FP_STR, OP_INC));

  op2_gemv(this, false, 1.0, DGConstants::DR, v, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DS, v, 0.0, op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::DT, v, 0.0, op_tmp[2]);

  op_par_loop(div_3d, "div_3d", cells,
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[2], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ry, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sy, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(ty, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(res, -1, OP_ID, DG_NP, DG_FP_STR, OP_INC));

  op2_gemv(this, false, 1.0, DGConstants::DR, w, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DS, w, 0.0, op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::DT, w, 0.0, op_tmp[2]);

  op_par_loop(div_3d, "div_3d", cells,
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[2], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(res, -1, OP_ID, DG_NP, DG_FP_STR, OP_INC));
}

void DGMesh3D::div_with_central_flux(op_dat u, op_dat v, op_dat w, op_dat res) {
  div(u, v, w, res);

  op_par_loop(zero_npf, "zero_npf", cells,
              op_arg_dat(op_tmp_npf[0], -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));

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
              op_arg_dat(op_tmp_npf[0], -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC));

  op2_gemv(this, false, 1.0, DGConstants::LIFT, op_tmp_npf[0], 1.0, res);
}

void DGMesh3D::div_weak(op_dat u, op_dat v, op_dat w, op_dat res) {
  op2_gemv(this, false, 1.0, DGConstants::DRW, u, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DSW, u, 0.0, op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::DTW, u, 0.0, op_tmp[2]);

  op_par_loop(zero_np, "zero_np", cells,
              op_arg_dat(res, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  op_par_loop(div_3d, "div_3d", cells,
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[2], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(res, -1, OP_ID, DG_NP, DG_FP_STR, OP_INC));

  op2_gemv(this, false, 1.0, DGConstants::DRW, v, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DSW, v, 0.0, op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::DTW, v, 0.0, op_tmp[2]);

  op_par_loop(div_3d, "div_3d", cells,
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[2], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ry, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sy, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(ty, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(res, -1, OP_ID, DG_NP, DG_FP_STR, OP_INC));

  op2_gemv(this, false, 1.0, DGConstants::DRW, w, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DSW, w, 0.0, op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::DTW, w, 0.0, op_tmp[2]);

  op_par_loop(div_3d, "div_3d", cells,
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[2], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(res, -1, OP_ID, DG_NP, DG_FP_STR, OP_INC));
}

void DGMesh3D::div_weak_with_central_flux(op_dat u, op_dat v, op_dat w, op_dat res) {
  div_weak(u, v, w, res);

  op_par_loop(zero_npf, "zero_npf", cells,
              op_arg_dat(op_tmp_npf[0], -1, OP_ID, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE));

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
              op_arg_dat(op_tmp_npf[0], -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_INC));

  // TODO bfaces
  op2_gemv(this, false, 1.0, DGConstants::LIFT, op_tmp_npf[0], -1.0, res);
}

void DGMesh3D::curl(op_dat u, op_dat v, op_dat w,
          op_dat resx, op_dat resy, op_dat resz) {
  op2_gemv(this, false, 1.0, DGConstants::DR, u, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DS, u, 0.0, op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::DT, u, 0.0, op_tmp[2]);

  op_par_loop(curl0_3d, "curl0_3d", cells,
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[2], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ry, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sy, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(ty, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(rz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(resy, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(resz, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::DR, v, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DS, v, 0.0, op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::DT, v, 0.0, op_tmp[2]);

  op_par_loop(curl1_3d, "curl1_3d", cells,
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[2], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(rz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(resx, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(resz, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));

  op2_gemv(this, false, 1.0, DGConstants::DR, w, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DS, w, 0.0, op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::DT, w, 0.0, op_tmp[2]);

  op_par_loop(curl2_3d, "curl2_3d", cells,
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[2], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(ry, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sy, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(ty, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(resx, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(resy, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
}

void DGMesh3D::mass(op_dat u) {
  op_par_loop(mass, "mass", cells,
              op_arg_gbl(constants->get_mat_ptr(DGConstants::MASS), DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(J, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(u, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
}

void DGMesh3D::inv_mass(op_dat u) {
  op_par_loop(inv_mass, "inv_mass", cells,
              op_arg_gbl(constants->get_mat_ptr(DGConstants::INV_MASS), DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(J, -1, OP_ID, 1, DG_FP_STR, OP_READ),
              op_arg_dat(u, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
}
