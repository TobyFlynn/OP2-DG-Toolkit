#include "dg_mesh/dg_mesh_2d.h"

#include "op_seq.h"

#include "dg_compiler_defs.h"
#include "dg_op2_blas.h"
#include "dg_constants/dg_constants.h"

void DGMesh2D::div(op_dat u, op_dat v, op_dat res) {
  op2_gemv(this, false, 1.0, DGConstants::DR, u, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DS, u, 0.0, op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::DR, v, 0.0, op_tmp[2]);
  op2_gemv(this, false, 1.0, DGConstants::DS, v, 0.0, op_tmp[3]);

  op_par_loop(div_2d, "div_2d", cells,
              op_arg_dat(order,     -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(op_tmp[2], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(op_tmp[3], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(rx,  -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(sx,  -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(ry,  -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(sy,  -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(res, -1, OP_ID, DG_NP, "double", OP_WRITE));
}

void DGMesh2D::div_weak(op_dat u, op_dat v, op_dat res) {
  op2_gemv(this, false, 1.0, DGConstants::DRW, u, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DSW, u, 0.0, op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::DRW, v, 0.0, op_tmp[2]);
  op2_gemv(this, false, 1.0, DGConstants::DSW, v, 0.0, op_tmp[3]);

  op_par_loop(div_2d, "div_2d", cells,
              op_arg_dat(order,     -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(op_tmp[2], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(op_tmp[3], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(rx, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(sx, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(ry, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(sy, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(res, -1, OP_ID, DG_NP, "double", OP_WRITE));
}

void DGMesh2D::div_with_central_flux(op_dat u, op_dat v, op_dat res) {
  div(u, v, res);

  // Central flux
  op2_gemv(this, false, 1.0, DGConstants::GAUSS_INTERP, u, 0.0, gauss->op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::GAUSS_INTERP, v, 0.0, gauss->op_tmp[1]);

  op_par_loop(zero_g_np, "zero_g_np", cells,
              op_arg_dat(gauss->op_tmp[2], -1, OP_ID, DG_G_NP, "double", OP_WRITE));
  
  op_par_loop(div_2d_central_flux, "div_2d_central_flux", edges,
              op_arg_dat(edgeNum,   -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(reverse,   -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(gauss->nx, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->ny, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->sJ, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->op_tmp[0], -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->op_tmp[1], -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->op_tmp[2], -2, edge2cells, DG_G_NP, "double", OP_INC));
  
  op2_gemv(this, false, -1.0, DGConstants::INV_MASS_GAUSS_INTERP_T, gauss->op_tmp[2], 1.0, res);
}

void DGMesh2D::curl(op_dat u, op_dat v, op_dat res) {
  // Same matrix multiplications as div
  op2_gemv(this, false, 1.0, DGConstants::DR, u, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DS, u, 0.0, op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::DR, v, 0.0, op_tmp[2]);
  op2_gemv(this, false, 1.0, DGConstants::DS, v, 0.0, op_tmp[3]);

  op_par_loop(curl_2d, "curl_2d", cells,
              op_arg_dat(order,     -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(op_tmp[2], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(op_tmp[3], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(rx,  -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(sx,  -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(ry,  -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(sy,  -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(res, -1, OP_ID, DG_NP, "double", OP_WRITE));
}

void DGMesh2D::grad(op_dat u, op_dat ux, op_dat uy) {
  op2_gemv(this, false, 1.0, DGConstants::DR, u, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DS, u, 0.0, op_tmp[1]);

  op_par_loop(grad_2d, "grad_2d", cells,
              op_arg_dat(order,     -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(rx, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(sx, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(ry, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(sy, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(ux, -1, OP_ID, DG_NP, "double", OP_WRITE),
              op_arg_dat(uy, -1, OP_ID, DG_NP, "double", OP_WRITE));
}

void DGMesh2D::grad_with_central_flux(op_dat u, op_dat ux, op_dat uy) {
  grad(u, ux, uy);

  // Central flux
  op2_gemv(this, false, 1.0, DGConstants::GAUSS_INTERP, u, 0.0, gauss->op_tmp[0]);

  op_par_loop(zero_g_np_2, "zero_g_np_2", cells,
              op_arg_dat(gauss->op_tmp[1], -1, OP_ID, DG_G_NP, "double", OP_WRITE),
              op_arg_dat(gauss->op_tmp[2], -1, OP_ID, DG_G_NP, "double", OP_WRITE));
  
  op_par_loop(grad_2d_central_flux, "grad_2d_central_flux", edges,
              op_arg_dat(edgeNum,   -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(reverse,   -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(gauss->nx, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->ny, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->sJ, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->op_tmp[0], -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->op_tmp[1], -2, edge2cells, DG_G_NP, "double", OP_INC),
              op_arg_dat(gauss->op_tmp[2], -2, edge2cells, DG_G_NP, "double", OP_INC));
  
  op2_gemv(this, false, -1.0, DGConstants::INV_MASS_GAUSS_INTERP_T, gauss->op_tmp[1], 1.0, ux);
  op2_gemv(this, false, -1.0, DGConstants::INV_MASS_GAUSS_INTERP_T, gauss->op_tmp[2], 1.0, uy);
}

void DGMesh2D::cub_grad(op_dat u, op_dat ux, op_dat uy) {
  op2_gemv(this, false, 1.0, DGConstants::CUB_DR, u, 0.0, cubature->op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::CUB_DS, u, 0.0, cubature->op_tmp[1]);

  op_par_loop(cub_grad_2d, "cub_grad_2d", cells,
              op_arg_dat(order,        -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(cubature->rx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->sx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->ry, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->sy, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->J,  -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->op_tmp[0], -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(cubature->op_tmp[1], -1, OP_ID, DG_CUB_NP, "double", OP_RW));

  op2_gemv(this, true, 1.0, DGConstants::CUB_V, cubature->op_tmp[0], 0.0, ux);
  op2_gemv(this, true, 1.0, DGConstants::CUB_V, cubature->op_tmp[1], 0.0, uy);
}

void DGMesh2D::cub_grad_with_central_flux(op_dat u, op_dat ux, op_dat uy) {
  cub_grad(u, ux, uy);

  // Central flux
  op2_gemv(this, false, 1.0, DGConstants::GAUSS_INTERP, u, 0.0, gauss->op_tmp[0]);

  op_par_loop(zero_g_np_2, "zero_g_np_2", cells,
              op_arg_dat(gauss->op_tmp[1], -1, OP_ID, DG_G_NP, "double", OP_WRITE),
              op_arg_dat(gauss->op_tmp[2], -1, OP_ID, DG_G_NP, "double", OP_WRITE));
  
  op_par_loop(grad_2d_central_flux, "grad_2d_central_flux", edges,
              op_arg_dat(edgeNum,   -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(reverse,   -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(gauss->nx, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->ny, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->sJ, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->op_tmp[0], -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->op_tmp[1], -2, edge2cells, DG_G_NP, "double", OP_INC),
              op_arg_dat(gauss->op_tmp[2], -2, edge2cells, DG_G_NP, "double", OP_INC));
  
  op2_gemv(this, true, -1.0, DGConstants::GAUSS_INTERP, gauss->op_tmp[1], 1.0, ux);
  op2_gemv(this, true, -1.0, DGConstants::GAUSS_INTERP, gauss->op_tmp[2], 1.0, uy);

  inv_mass(ux);
  inv_mass(uy);
}

void DGMesh2D::cub_div(op_dat u, op_dat v, op_dat res) {
  op2_gemv(this, false, 1.0, DGConstants::CUB_DR, u, 0.0, cubature->op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::CUB_DS, u, 0.0, cubature->op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::CUB_DR, v, 0.0, cubature->op_tmp[2]);
  op2_gemv(this, false, 1.0, DGConstants::CUB_DS, v, 0.0, cubature->op_tmp[3]);

  op_par_loop(cub_div_2d, "cub_div_2d", cells,
              op_arg_dat(order,        -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(cubature->rx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->sx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->ry, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->sy, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->J, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->op_tmp[0], -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(cubature->op_tmp[1], -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->op_tmp[2], -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->op_tmp[3], -1, OP_ID, DG_CUB_NP, "double", OP_READ));

  op2_gemv(this, true, 1.0, DGConstants::CUB_V, cubature->op_tmp[0], 0.0, res);
}

void DGMesh2D::cub_div_with_central_flux_no_inv_mass(op_dat u, op_dat v, op_dat res) {
  cub_div(u, v, res);

  // Central flux
  op2_gemv(this, false, 1.0, DGConstants::GAUSS_INTERP, u, 0.0, gauss->op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::GAUSS_INTERP, v, 0.0, gauss->op_tmp[1]);

  op_par_loop(zero_g_np, "zero_g_np", cells,
              op_arg_dat(gauss->op_tmp[2], -1, OP_ID, DG_G_NP, "double", OP_WRITE));
  
  op_par_loop(div_2d_central_flux, "div_2d_central_flux", edges,
              op_arg_dat(edgeNum,   -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(reverse,   -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(gauss->nx, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->ny, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->sJ, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->op_tmp[0], -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->op_tmp[1], -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->op_tmp[2], -2, edge2cells, DG_G_NP, "double", OP_INC));
  
  op2_gemv(this, true, -1.0, DGConstants::GAUSS_INTERP, gauss->op_tmp[2], 1.0, res);
}

void DGMesh2D::cub_div_with_central_flux(op_dat u, op_dat v, op_dat res) {
  cub_div_with_central_flux_no_inv_mass(u, v, res);

  inv_mass(res);
}

void DGMesh2D::cub_grad_weak(op_dat u, op_dat ux, op_dat uy) {
  op2_gemv(this, false, 1.0, DGConstants::CUB_V, u, 0.0, cubature->op_tmp[0]);

  op_par_loop(cub_grad_weak_2d, "cub_grad_weak_2d", cells,
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(cubature->op_tmp[0], -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(cubature->rx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->sx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->ry, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->sy, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->J,  -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->op_tmp[1], -1, OP_ID, DG_CUB_NP, "double", OP_WRITE),
              op_arg_dat(cubature->op_tmp[2], -1, OP_ID, DG_CUB_NP, "double", OP_WRITE),
              op_arg_dat(cubature->op_tmp[3], -1, OP_ID, DG_CUB_NP, "double", OP_WRITE));

  op2_gemv(this, true, 1.0, DGConstants::CUB_DR, cubature->op_tmp[0], 0.0, ux);
  op2_gemv(this, true, 1.0, DGConstants::CUB_DS, cubature->op_tmp[1], 1.0, ux);
  op2_gemv(this, true, 1.0, DGConstants::CUB_DR, cubature->op_tmp[2], 0.0, uy);
  op2_gemv(this, true, 1.0, DGConstants::CUB_DS, cubature->op_tmp[3], 1.0, uy);
}

void DGMesh2D::cub_grad_weak_with_central_flux(op_dat u, op_dat ux, op_dat uy) {
  cub_grad_weak(u, ux, uy);

  // Central flux
  op2_gemv(this, false, 1.0, DGConstants::GAUSS_INTERP, u, 0.0, gauss->op_tmp[0]);

  op_par_loop(zero_g_np_2, "zero_g_np_2", cells,
              op_arg_dat(gauss->op_tmp[1], -1, OP_ID, DG_G_NP, "double", OP_WRITE),
              op_arg_dat(gauss->op_tmp[2], -1, OP_ID, DG_G_NP, "double", OP_WRITE));
  
  op_par_loop(grad_weak_2d_central_flux, "grad_weak_2d_central_flux", edges,
              op_arg_dat(edgeNum,   -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(reverse,   -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(gauss->nx, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->ny, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->sJ, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->op_tmp[0], -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->op_tmp[1], -2, edge2cells, DG_G_NP, "double", OP_INC),
              op_arg_dat(gauss->op_tmp[2], -2, edge2cells, DG_G_NP, "double", OP_INC));

  op2_gemv(this, true, 1.0, DGConstants::GAUSS_INTERP, gauss->op_tmp[1], -1.0, ux);
  op2_gemv(this, true, 1.0, DGConstants::GAUSS_INTERP, gauss->op_tmp[2], -1.0, uy);

  inv_mass(ux);
  inv_mass(uy);
}

void DGMesh2D::cub_div_weak(op_dat u, op_dat v, op_dat res) {
  op2_gemv(this, false, 1.0, DGConstants::CUB_V, u, 0.0, cubature->op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::CUB_V, v, 0.0, cubature->op_tmp[1]);

  op_par_loop(cub_div_weak_2d, "cub_div_weak_2d", cells,
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(cubature->op_tmp[0], -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(cubature->op_tmp[1], -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(cubature->rx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->sx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->ry, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->sy, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->J,  -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(cubature->op_tmp[2], -1, OP_ID, DG_CUB_NP, "double", OP_WRITE),
              op_arg_dat(cubature->op_tmp[3], -1, OP_ID, DG_CUB_NP, "double", OP_WRITE));

  op2_gemv(this, true, 1.0, DGConstants::CUB_DR, cubature->op_tmp[0], 0.0, res);
  op2_gemv(this, true, 1.0, DGConstants::CUB_DS, cubature->op_tmp[1], 1.0, res);
  op2_gemv(this, true, 1.0, DGConstants::CUB_DR, cubature->op_tmp[2], 1.0, res);
  op2_gemv(this, true, 1.0, DGConstants::CUB_DS, cubature->op_tmp[3], 1.0, res);
}

void DGMesh2D::cub_div_weak_with_central_flux(op_dat u, op_dat v, op_dat res) {
  cub_div_weak(u, v, res);

  // Central flux
  op2_gemv(this, false, 1.0, DGConstants::GAUSS_INTERP, u, 0.0, gauss->op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::GAUSS_INTERP, v, 0.0, gauss->op_tmp[1]);

  op_par_loop(zero_g_np, "zero_g_np", cells,
              op_arg_dat(gauss->op_tmp[2], -1, OP_ID, DG_G_NP, "double", OP_WRITE));
  
  op_par_loop(div_weak_2d_central_flux, "div_weak_2d_central_flux", edges,
              op_arg_dat(edgeNum,   -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(reverse,   -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(gauss->nx, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->ny, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->sJ, -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->op_tmp[0], -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->op_tmp[1], -2, edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(gauss->op_tmp[2], -2, edge2cells, DG_G_NP, "double", OP_INC));
  
  op2_gemv(this, true, 1.0, DGConstants::GAUSS_INTERP, gauss->op_tmp[2], -1.0, res);

  inv_mass(res);
}

void DGMesh2D::inv_mass(op_dat u) {
  op_par_loop(inv_J, "inv_J", cells,
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(J, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(u, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, "double", OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::INV_MASS, op_tmp[0], 0.0, u);
}
