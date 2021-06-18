#include "operators.h"

#include "op_seq.h"
#include "dg_mesh.h"
#include "dg_blas_calls.h"

void div(DGMesh *mesh, op_dat u, op_dat v, op_dat res) {
  op2_gemv(true, 15, 15, 1.0, constants->get_ptr(DGConstants::DR), 15, u, 0.0, mesh->op_tmp[0]);
  op2_gemv(true, 15, 15, 1.0, constants->get_ptr(DGConstants::DS), 15, u, 0.0, mesh->op_tmp[1]);
  op2_gemv(true, 15, 15, 1.0, constants->get_ptr(DGConstants::DR), 15, v, 0.0, mesh->op_tmp[2]);
  op2_gemv(true, 15, 15, 1.0, constants->get_ptr(DGConstants::DS), 15, v, 0.0, mesh->op_tmp[3]);

  op_par_loop(div, "div", mesh->cells,
              op_arg_dat(mesh->op_tmp[0], -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[1], -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[2], -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[3], -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->rx, -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->sx, -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->ry, -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->sy, -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(res, -1, OP_ID, 15, "double", OP_WRITE));
}

void curl(DGMesh *mesh, op_dat u, op_dat v, op_dat res) {
  // Same matrix multiplications as div
  op2_gemv(true, 15, 15, 1.0, constants->get_ptr(DGConstants::DR), 15, u, 0.0, mesh->op_tmp[0]);
  op2_gemv(true, 15, 15, 1.0, constants->get_ptr(DGConstants::DS), 15, u, 0.0, mesh->op_tmp[1]);
  op2_gemv(true, 15, 15, 1.0, constants->get_ptr(DGConstants::DR), 15, v, 0.0, mesh->op_tmp[2]);
  op2_gemv(true, 15, 15, 1.0, constants->get_ptr(DGConstants::DS), 15, v, 0.0, mesh->op_tmp[3]);

  op_par_loop(curl, "curl", mesh->cells,
              op_arg_dat(mesh->op_tmp[0], -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[1], -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[2], -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[3], -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->rx, -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->sx, -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->ry, -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->sy, -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(res, -1, OP_ID, 15, "double", OP_WRITE));
}

void grad(DGMesh *mesh, op_dat u, op_dat ux, op_dat uy) {
  op2_gemv(true, 15, 15, 1.0, constants->get_ptr(DGConstants::DR), 15, u, 0.0, mesh->op_tmp[0]);
  op2_gemv(true, 15, 15, 1.0, constants->get_ptr(DGConstants::DS), 15, u, 0.0, mesh->op_tmp[1]);

  op_par_loop(grad, "grad", data->cells,
              op_arg_dat(mesh->op_tmp[0], -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->op_tmp[1], -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->rx, -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->sx, -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->ry, -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(mesh->sy, -1, OP_ID, 15, "double", OP_READ),
              op_arg_dat(ux, -1, OP_ID, 15, "double", OP_WRITE),
              op_arg_dat(uy, -1, OP_ID, 15, "double", OP_WRITE));
}
