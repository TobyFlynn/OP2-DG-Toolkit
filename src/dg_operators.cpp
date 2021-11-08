#include "dg_operators.h"

#include "op_seq.h"
#include "dg_mesh.h"
#include "dg_blas_calls.h"
#include "dg_compiler_defs.h"

void div(DGMesh *mesh, op_dat u, op_dat v, op_dat res) {
  op2_gemv(false, DG_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::DR), DG_NP, u, 0.0, mesh->op_tmp[0]);
  op2_gemv(false, DG_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::DS), DG_NP, u, 0.0, mesh->op_tmp[1]);
  op2_gemv(false, DG_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::DR), DG_NP, v, 0.0, mesh->op_tmp[2]);
  op2_gemv(false, DG_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::DS), DG_NP, v, 0.0, mesh->op_tmp[3]);

  op_par_loop(div, "div", mesh->cells,
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

void curl(DGMesh *mesh, op_dat u, op_dat v, op_dat res) {
  // Same matrix multiplications as div
  op2_gemv(false, DG_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::DR), DG_NP, u, 0.0, mesh->op_tmp[0]);
  op2_gemv(false, DG_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::DS), DG_NP, u, 0.0, mesh->op_tmp[1]);
  op2_gemv(false, DG_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::DR), DG_NP, v, 0.0, mesh->op_tmp[2]);
  op2_gemv(false, DG_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::DS), DG_NP, v, 0.0, mesh->op_tmp[3]);

  op_par_loop(curl, "curl", mesh->cells,
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
  op2_gemv(false, DG_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::DR), DG_NP, u, 0.0, mesh->op_tmp[0]);
  op2_gemv(false, DG_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::DS), DG_NP, u, 0.0, mesh->op_tmp[1]);

  op_par_loop(grad, "grad", mesh->cells,
              op_arg_dat(mesh->op_tmp[0], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[1], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->rx, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->sx, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->ry, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->sy, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(ux, -1, OP_ID, DG_NP, "double", OP_WRITE),
              op_arg_dat(uy, -1, OP_ID, DG_NP, "double", OP_WRITE));
}

void cub_grad(DGMesh *mesh, op_dat u, op_dat ux, op_dat uy) {
  op2_gemv(false, DG_CUB_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::CUB_DR), DG_CUB_NP, u, 0.0, mesh->cubature->op_tmp[0]);
  op2_gemv(false, DG_CUB_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::CUB_DS), DG_CUB_NP, u, 0.0, mesh->cubature->op_tmp[1]);

  op_par_loop(cub_grad, "cub_grad", mesh->cells,
              op_arg_dat(mesh->cubature->rx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->sx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->ry, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->sy, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->J, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->op_tmp[0], -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(mesh->cubature->op_tmp[1], -1, OP_ID, DG_CUB_NP, "double", OP_RW));

  op2_gemv(true, DG_NP, DG_CUB_NP, 1.0, constants->get_ptr(DGConstants::CUB_V), DG_CUB_NP, mesh->cubature->op_tmp[0], 0.0, ux);
  op2_gemv(true, DG_NP, DG_CUB_NP, 1.0, constants->get_ptr(DGConstants::CUB_V), DG_CUB_NP, mesh->cubature->op_tmp[1], 0.0, uy);
}

void cub_div(DGMesh *mesh, op_dat u, op_dat v, op_dat res) {
  op2_gemv(false, DG_CUB_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::CUB_DR), DG_CUB_NP, u, 0.0, mesh->cubature->op_tmp[0]);
  op2_gemv(false, DG_CUB_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::CUB_DS), DG_CUB_NP, u, 0.0, mesh->cubature->op_tmp[1]);
  op2_gemv(false, DG_CUB_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::CUB_DR), DG_CUB_NP, v, 0.0, mesh->cubature->op_tmp[2]);
  op2_gemv(false, DG_CUB_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::CUB_DS), DG_CUB_NP, v, 0.0, mesh->cubature->op_tmp[3]);

  op_par_loop(cub_div, "cub_div", mesh->cells,
                op_arg_dat(mesh->cubature->rx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
                op_arg_dat(mesh->cubature->sx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
                op_arg_dat(mesh->cubature->ry, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
                op_arg_dat(mesh->cubature->sy, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
                op_arg_dat(mesh->cubature->J, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
                op_arg_dat(mesh->cubature->op_tmp[0], -1, OP_ID, DG_CUB_NP, "double", OP_RW),
                op_arg_dat(mesh->cubature->op_tmp[1], -1, OP_ID, DG_CUB_NP, "double", OP_READ),
                op_arg_dat(mesh->cubature->op_tmp[2], -1, OP_ID, DG_CUB_NP, "double", OP_READ),
                op_arg_dat(mesh->cubature->op_tmp[3], -1, OP_ID, DG_CUB_NP, "double", OP_READ));

  op2_gemv(true, DG_NP, DG_CUB_NP, 1.0, constants->get_ptr(DGConstants::CUB_V), DG_CUB_NP, mesh->cubature->op_tmp[0], 0.0, res);
}

void cub_grad_weak(DGMesh *mesh, op_dat u, op_dat ux, op_dat uy) {
  op2_gemv(false, DG_CUB_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::CUB_V), DG_CUB_NP, u, 0.0, mesh->cubature->op_tmp[0]);

  op_par_loop(cub_grad_weak, "cub_grad_weak", mesh->cells,
              op_arg_dat(mesh->cubature->op_tmp[0], -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(mesh->cubature->rx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->sx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->ry, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->sy, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->J, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->op_tmp[1], -1, OP_ID, DG_CUB_NP, "double", OP_WRITE),
              op_arg_dat(mesh->cubature->op_tmp[2], -1, OP_ID, DG_CUB_NP, "double", OP_WRITE),
              op_arg_dat(mesh->cubature->op_tmp[3], -1, OP_ID, DG_CUB_NP, "double", OP_WRITE));

  op2_gemv(true, DG_NP, DG_CUB_NP, 1.0, constants->get_ptr(DGConstants::CUB_DR), DG_CUB_NP, mesh->cubature->op_tmp[0], 0.0, ux);
  op2_gemv(true, DG_NP, DG_CUB_NP, 1.0, constants->get_ptr(DGConstants::CUB_DS), DG_CUB_NP, mesh->cubature->op_tmp[1], 1.0, ux);
  op2_gemv(true, DG_NP, DG_CUB_NP, 1.0, constants->get_ptr(DGConstants::CUB_DR), DG_CUB_NP, mesh->cubature->op_tmp[2], 0.0, uy);
  op2_gemv(true, DG_NP, DG_CUB_NP, 1.0, constants->get_ptr(DGConstants::CUB_DS), DG_CUB_NP, mesh->cubature->op_tmp[3], 1.0, uy);
}

void cub_div_weak(DGMesh *mesh, op_dat u, op_dat v, op_dat res) {
  op2_gemv(false, DG_CUB_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::CUB_V), DG_CUB_NP, u, 0.0, mesh->cubature->op_tmp[0]);
  op2_gemv(false, DG_CUB_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::CUB_V), DG_CUB_NP, v, 0.0, mesh->cubature->op_tmp[1]);

  op_par_loop(cub_div_weak, "cub_div_weak", mesh->cells,
              op_arg_dat(mesh->cubature->op_tmp[0], -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(mesh->cubature->op_tmp[1], -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(mesh->cubature->rx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->sx, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->ry, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->sy, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->J, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
              op_arg_dat(mesh->cubature->op_tmp[2], -1, OP_ID, DG_CUB_NP, "double", OP_WRITE),
              op_arg_dat(mesh->cubature->op_tmp[3], -1, OP_ID, DG_CUB_NP, "double", OP_WRITE));

  op2_gemv(true, DG_NP, DG_CUB_NP, 1.0, constants->get_ptr(DGConstants::CUB_DR), DG_CUB_NP, mesh->cubature->op_tmp[0], 0.0, res);
  op2_gemv(true, DG_NP, DG_CUB_NP, 1.0, constants->get_ptr(DGConstants::CUB_DS), DG_CUB_NP, mesh->cubature->op_tmp[1], 1.0, res);
  op2_gemv(true, DG_NP, DG_CUB_NP, 1.0, constants->get_ptr(DGConstants::CUB_DR), DG_CUB_NP, mesh->cubature->op_tmp[2], 1.0, res);
  op2_gemv(true, DG_NP, DG_CUB_NP, 1.0, constants->get_ptr(DGConstants::CUB_DS), DG_CUB_NP, mesh->cubature->op_tmp[3], 1.0, res);
}

void inv_mass(DGMesh *mesh, op_dat u) {
  op2_gemv(false, DG_NP, DG_NP, 1.0, constants->get_ptr(DGConstants::INV_MASS), DG_NP, u, 0.0, mesh->op_tmp[0]);

  op_par_loop(inv_J, "inv_J", mesh->cells,
              op_arg_dat(mesh->J, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[0], -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(u, -1, OP_ID, DG_NP, "double", OP_WRITE));
}
