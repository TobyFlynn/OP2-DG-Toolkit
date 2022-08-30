#include "dg_operators.h"

#include "op_seq.h"
#include "dg_mesh.h"
#include "dg_blas_calls.h"
#include "dg_compiler_defs.h"
#include "dg_op2_blas.h"

void div(DGMesh *mesh, op_dat u, op_dat v, op_dat res) {
  op2_gemv(mesh, false, 1.0, DGConstants::DR, u, 0.0, mesh->op_tmp[0]);
  op2_gemv(mesh, false, 1.0, DGConstants::DS, u, 0.0, mesh->op_tmp[1]);
  op2_gemv(mesh, false, 1.0, DGConstants::DR, v, 0.0, mesh->op_tmp[2]);
  op2_gemv(mesh, false, 1.0, DGConstants::DS, v, 0.0, mesh->op_tmp[3]);

  op_par_loop(div, "div", mesh->cells,
              op_arg_dat(mesh->order,     -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(mesh->op_tmp[0], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[1], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[2], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[3], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->rx, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->sx, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->ry, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->sy, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(res, -1, OP_ID, DG_NP, "double", OP_WRITE));
}

void div_with_central_flux(DGMesh *mesh, op_dat u, op_dat v, op_dat res) {
  div(mesh, u, v, res);

  // Central flux
  op2_gemv(mesh, false, 1.0, DGConstants::GAUSS_INTERP, u, 0.0, mesh->gauss->op_tmp[0]);
  op2_gemv(mesh, false, 1.0, DGConstants::GAUSS_INTERP, v, 0.0, mesh->gauss->op_tmp[1]);

  op_par_loop(zero_g_np, "zero_g_np", mesh->cells,
              op_arg_dat(mesh->gauss->op_tmp[2], -1, OP_ID, DG_G_NP, "double", OP_WRITE));
  
  op_par_loop(div_central_flux, "div_central_flux", mesh->edges,
              op_arg_dat(mesh->edgeNum,   -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(mesh->reverse,   -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(mesh->gauss->nx, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->ny, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->sJ, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->op_tmp[0], -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->op_tmp[1], -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->op_tmp[2], -2, mesh->edge2cells, DG_G_NP, "double", OP_INC));
  
  op2_gemv(mesh, false, -1.0, DGConstants::INV_MASS_GAUSS_INTERP_T, mesh->gauss->op_tmp[2], 1.0, res);
}

void curl(DGMesh *mesh, op_dat u, op_dat v, op_dat res) {
  // Same matrix multiplications as div
  op2_gemv(mesh, false, 1.0, DGConstants::DR, u, 0.0, mesh->op_tmp[0]);
  op2_gemv(mesh, false, 1.0, DGConstants::DS, u, 0.0, mesh->op_tmp[1]);
  op2_gemv(mesh, false, 1.0, DGConstants::DR, v, 0.0, mesh->op_tmp[2]);
  op2_gemv(mesh, false, 1.0, DGConstants::DS, v, 0.0, mesh->op_tmp[3]);

  op_par_loop(curl, "curl", mesh->cells,
              op_arg_dat(mesh->order,     -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(mesh->op_tmp[0], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[1], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[2], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[3], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->rx, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->sx, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->ry, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->sy, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(res, -1, OP_ID, DG_NP, "double", OP_WRITE));
}

void grad(DGMesh *mesh, op_dat u, op_dat ux, op_dat uy) {
  op2_gemv(mesh, false, 1.0, DGConstants::DR, u, 0.0, mesh->op_tmp[0]);
  op2_gemv(mesh, false, 1.0, DGConstants::DS, u, 0.0, mesh->op_tmp[1]);

  op_par_loop(grad, "grad", mesh->cells,
              op_arg_dat(mesh->order,     -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(mesh->op_tmp[0], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[1], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->rx, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->sx, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->ry, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->sy, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(ux, -1, OP_ID, DG_NP, "double", OP_WRITE),
              op_arg_dat(uy, -1, OP_ID, DG_NP, "double", OP_WRITE));
}

void grad_with_central_flux(DGMesh *mesh, op_dat u, op_dat ux, op_dat uy) {
  grad(mesh, u, ux, uy);

  // Central flux
  op2_gemv(mesh, false, 1.0, DGConstants::GAUSS_INTERP, u, 0.0, mesh->gauss->op_tmp[0]);

  op_par_loop(zero_g_np_2, "zero_g_np_2", mesh->cells,
              op_arg_dat(mesh->gauss->op_tmp[1], -1, OP_ID, DG_G_NP, "double", OP_WRITE),
              op_arg_dat(mesh->gauss->op_tmp[2], -1, OP_ID, DG_G_NP, "double", OP_WRITE));
  
  op_par_loop(grad_central_flux, "grad_central_flux", mesh->edges,
              op_arg_dat(mesh->edgeNum,   -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(mesh->reverse,   -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(mesh->gauss->nx, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->ny, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->sJ, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->op_tmp[0], -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->op_tmp[1], -2, mesh->edge2cells, DG_G_NP, "double", OP_INC),
              op_arg_dat(mesh->gauss->op_tmp[2], -2, mesh->edge2cells, DG_G_NP, "double", OP_INC));
  
  op2_gemv(mesh, false, -1.0, DGConstants::INV_MASS_GAUSS_INTERP_T, mesh->gauss->op_tmp[1], 1.0, ux);
  op2_gemv(mesh, false, -1.0, DGConstants::INV_MASS_GAUSS_INTERP_T, mesh->gauss->op_tmp[2], 1.0, uy);
}

void cub_grad(DGMesh *mesh, op_dat u, op_dat ux, op_dat uy) {
  op2_gemv(mesh, false, 1.0, DGConstants::CUB_DR, u, 0.0, mesh->cubature->op_tmp[0]);
  op2_gemv(mesh, false, 1.0, DGConstants::CUB_DS, u, 0.0, mesh->cubature->op_tmp[1]);

  op_par_loop(cub_grad, "cub_grad", mesh->cells,
              op_arg_dat(mesh->order,        -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(mesh->cubature->rx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->sx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->ry, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->sy, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->J,  -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->op_tmp[0], -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(mesh->cubature->op_tmp[1], -1, OP_ID, DG_CUB_NP, "double", OP_RW));

  op2_gemv(mesh, true, 1.0, DGConstants::CUB_V, mesh->cubature->op_tmp[0], 0.0, ux);
  op2_gemv(mesh, true, 1.0, DGConstants::CUB_V, mesh->cubature->op_tmp[1], 0.0, uy);
}

void cub_grad_with_central_flux(DGMesh *mesh, op_dat u, op_dat ux, op_dat uy) {
  cub_grad(mesh, u, ux, uy);

  // Central flux
  op2_gemv(mesh, false, 1.0, DGConstants::GAUSS_INTERP, u, 0.0, mesh->gauss->op_tmp[0]);

  op_par_loop(zero_g_np_2, "zero_g_np_2", mesh->cells,
              op_arg_dat(mesh->gauss->op_tmp[1], -1, OP_ID, DG_G_NP, "double", OP_WRITE),
              op_arg_dat(mesh->gauss->op_tmp[2], -1, OP_ID, DG_G_NP, "double", OP_WRITE));
  
  op_par_loop(grad_central_flux, "grad_central_flux", mesh->edges,
              op_arg_dat(mesh->edgeNum,   -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(mesh->reverse,   -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(mesh->gauss->nx, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->ny, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->sJ, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->op_tmp[0], -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->op_tmp[1], -2, mesh->edge2cells, DG_G_NP, "double", OP_INC),
              op_arg_dat(mesh->gauss->op_tmp[2], -2, mesh->edge2cells, DG_G_NP, "double", OP_INC));
  
  op2_gemv(mesh, true, -1.0, DGConstants::GAUSS_INTERP, mesh->gauss->op_tmp[1], 1.0, ux);
  op2_gemv(mesh, true, -1.0, DGConstants::GAUSS_INTERP, mesh->gauss->op_tmp[2], 1.0, uy);

  inv_mass(mesh, ux);
  inv_mass(mesh, uy);
}

void cub_div(DGMesh *mesh, op_dat u, op_dat v, op_dat res) {
  op2_gemv(mesh, false, 1.0, DGConstants::CUB_DR, u, 0.0, mesh->cubature->op_tmp[0]);
  op2_gemv(mesh, false, 1.0, DGConstants::CUB_DS, u, 0.0, mesh->cubature->op_tmp[1]);
  op2_gemv(mesh, false, 1.0, DGConstants::CUB_DR, v, 0.0, mesh->cubature->op_tmp[2]);
  op2_gemv(mesh, false, 1.0, DGConstants::CUB_DS, v, 0.0, mesh->cubature->op_tmp[3]);

  op_par_loop(cub_div, "cub_div", mesh->cells,
              op_arg_dat(mesh->order,        -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(mesh->cubature->rx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->sx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->ry, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->sy, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->J, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->op_tmp[0], -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(mesh->cubature->op_tmp[1], -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->op_tmp[2], -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->op_tmp[3], -1, OP_ID, DG_CUB_NP, "double", OP_READ));

  op2_gemv(mesh, true, 1.0, DGConstants::CUB_V, mesh->cubature->op_tmp[0], 0.0, res);
}

void cub_div_with_central_flux(DGMesh *mesh, op_dat u, op_dat v, op_dat res) {
  cub_div(mesh, u, v, res);

  // Central flux
  op2_gemv(mesh, false, 1.0, DGConstants::GAUSS_INTERP, u, 0.0, mesh->gauss->op_tmp[0]);
  op2_gemv(mesh, false, 1.0, DGConstants::GAUSS_INTERP, v, 0.0, mesh->gauss->op_tmp[1]);

  op_par_loop(zero_g_np, "zero_g_np", mesh->cells,
              op_arg_dat(mesh->gauss->op_tmp[2], -1, OP_ID, DG_G_NP, "double", OP_WRITE));
  
  op_par_loop(div_central_flux, "div_central_flux", mesh->edges,
              op_arg_dat(mesh->edgeNum,   -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(mesh->reverse,   -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(mesh->gauss->nx, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->ny, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->sJ, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->op_tmp[0], -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->op_tmp[1], -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->op_tmp[2], -2, mesh->edge2cells, DG_G_NP, "double", OP_INC));
  
  op2_gemv(mesh, true, -1.0, DGConstants::GAUSS_INTERP, mesh->gauss->op_tmp[2], 1.0, res);

  inv_mass(mesh, res);
}

void cub_grad_weak(DGMesh *mesh, op_dat u, op_dat ux, op_dat uy) {
  op2_gemv(mesh, false, 1.0, DGConstants::CUB_V, u, 0.0, mesh->cubature->op_tmp[0]);

  op_par_loop(cub_grad_weak, "cub_grad_weak", mesh->cells,
              op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(mesh->cubature->op_tmp[0], -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(mesh->cubature->rx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->sx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->ry, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->sy, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->J,  -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->op_tmp[1], -1, OP_ID, DG_CUB_NP, "double", OP_WRITE),
              op_arg_dat(mesh->cubature->op_tmp[2], -1, OP_ID, DG_CUB_NP, "double", OP_WRITE),
              op_arg_dat(mesh->cubature->op_tmp[3], -1, OP_ID, DG_CUB_NP, "double", OP_WRITE));

  op2_gemv(mesh, true, 1.0, DGConstants::CUB_DR, mesh->cubature->op_tmp[0], 0.0, ux);
  op2_gemv(mesh, true, 1.0, DGConstants::CUB_DS, mesh->cubature->op_tmp[1], 1.0, ux);
  op2_gemv(mesh, true, 1.0, DGConstants::CUB_DR, mesh->cubature->op_tmp[2], 0.0, uy);
  op2_gemv(mesh, true, 1.0, DGConstants::CUB_DS, mesh->cubature->op_tmp[3], 1.0, uy);
}

void cub_grad_weak_with_central_flux(DGMesh *mesh, op_dat u, op_dat ux, op_dat uy) {
  cub_grad_weak(mesh, u, ux, uy);

  // Central flux
  op2_gemv(mesh, false, 1.0, DGConstants::GAUSS_INTERP, u, 0.0, mesh->gauss->op_tmp[0]);

  op_par_loop(zero_g_np_2, "zero_g_np_2", mesh->cells,
              op_arg_dat(mesh->gauss->op_tmp[1], -1, OP_ID, DG_G_NP, "double", OP_WRITE),
              op_arg_dat(mesh->gauss->op_tmp[2], -1, OP_ID, DG_G_NP, "double", OP_WRITE));
  
  op_par_loop(grad_weak_central_flux, "grad_weak_central_flux", mesh->edges,
              op_arg_dat(mesh->edgeNum,   -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(mesh->reverse,   -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(mesh->gauss->nx, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->ny, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->sJ, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->op_tmp[0], -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->op_tmp[1], -2, mesh->edge2cells, DG_G_NP, "double", OP_INC),
              op_arg_dat(mesh->gauss->op_tmp[2], -2, mesh->edge2cells, DG_G_NP, "double", OP_INC));

  op2_gemv(mesh, true, 1.0, DGConstants::GAUSS_INTERP, mesh->gauss->op_tmp[1], -1.0, ux);
  op2_gemv(mesh, true, 1.0, DGConstants::GAUSS_INTERP, mesh->gauss->op_tmp[2], -1.0, uy);

  inv_mass(mesh, ux);
  inv_mass(mesh, uy);
}

void cub_div_weak(DGMesh *mesh, op_dat u, op_dat v, op_dat res) {
  op2_gemv(mesh, false, 1.0, DGConstants::CUB_V, u, 0.0, mesh->cubature->op_tmp[0]);
  op2_gemv(mesh, false, 1.0, DGConstants::CUB_V, v, 0.0, mesh->cubature->op_tmp[1]);

  op_par_loop(cub_div_weak, "cub_div_weak", mesh->cells,
              op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(mesh->cubature->op_tmp[0], -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(mesh->cubature->op_tmp[1], -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(mesh->cubature->rx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->sx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->ry, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->sy, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->J,  -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->op_tmp[2], -1, OP_ID, DG_CUB_NP, "double", OP_WRITE),
              op_arg_dat(mesh->cubature->op_tmp[3], -1, OP_ID, DG_CUB_NP, "double", OP_WRITE));

  op2_gemv(mesh, true, 1.0, DGConstants::CUB_DR, mesh->cubature->op_tmp[0], 0.0, res);
  op2_gemv(mesh, true, 1.0, DGConstants::CUB_DS, mesh->cubature->op_tmp[1], 1.0, res);
  op2_gemv(mesh, true, 1.0, DGConstants::CUB_DR, mesh->cubature->op_tmp[2], 1.0, res);
  op2_gemv(mesh, true, 1.0, DGConstants::CUB_DS, mesh->cubature->op_tmp[3], 1.0, res);
}

void cub_div_weak_with_central_flux(DGMesh *mesh, op_dat u, op_dat v, op_dat res) {
  cub_div_weak(mesh, u, v, res);

  // Central flux
  op2_gemv(mesh, false, 1.0, DGConstants::GAUSS_INTERP, u, 0.0, mesh->gauss->op_tmp[0]);
  op2_gemv(mesh, false, 1.0, DGConstants::GAUSS_INTERP, v, 0.0, mesh->gauss->op_tmp[1]);

  op_par_loop(zero_g_np, "zero_g_np", mesh->cells,
              op_arg_dat(mesh->gauss->op_tmp[2], -1, OP_ID, DG_G_NP, "double", OP_WRITE));
  
  op_par_loop(div_weak_central_flux, "div_weak_central_flux", mesh->edges,
              op_arg_dat(mesh->edgeNum,   -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(mesh->reverse,   -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(mesh->gauss->nx, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->ny, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->sJ, -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->op_tmp[0], -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->op_tmp[1], -2, mesh->edge2cells, DG_G_NP, "double", OP_READ),
              op_arg_dat(mesh->gauss->op_tmp[2], -2, mesh->edge2cells, DG_G_NP, "double", OP_INC));
  
  op2_gemv(mesh, true, 1.0, DGConstants::GAUSS_INTERP, mesh->gauss->op_tmp[2], -1.0, res);

  inv_mass(mesh, res);
}

void inv_mass(DGMesh *mesh, op_dat u) {
  op_par_loop(inv_J, "inv_J", mesh->cells,
              op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(mesh->J, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(u, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[0], -1, OP_ID, DG_NP, "double", OP_WRITE));

  op2_gemv(mesh, false, 1.0, DGConstants::INV_MASS, mesh->op_tmp[0], 0.0, u);
}

void filter(DGMesh *mesh, op_dat u) {
  op2_gemv(mesh, false, 1.0, DGConstants::INV_V, u, 0.0, mesh->op_tmp[0]);

  op_par_loop(filter, "filter", mesh->cells,
              op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(v_g, DG_ORDER * DG_NP * DG_NP, "double", OP_READ),
              op_arg_gbl(mass_g, DG_ORDER * DG_NP * DG_NP, "double", OP_READ),
              op_arg_dat(mesh->J, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(u, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[0], -1, OP_ID, DG_NP, "double", OP_RW),
              op_arg_dat(mesh->shock, -1, OP_ID, DG_NP, "double", OP_WRITE));

  op2_gemv(mesh, false, 1.0, DGConstants::V, mesh->op_tmp[0], 0.0, u);
}