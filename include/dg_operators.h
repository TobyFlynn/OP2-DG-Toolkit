#ifndef __DG_OPERATORS_H
#define __DG_OPERATORS_H

#include "op_seq.h"
#include "dg_mesh.h"

void div(DGMesh *mesh, op_dat u, op_dat v, op_dat res);

void curl(DGMesh *mesh, op_dat u, op_dat v, op_dat res);

void grad(DGMesh *mesh, op_dat u, op_dat ux, op_dat uy);

void cub_grad(DGMesh *mesh, op_dat u, op_dat ux, op_dat uy);

void cub_div(DGMesh *mesh, op_dat u, op_dat v, op_dat res);

void cub_grad_weak(DGMesh *mesh, op_dat u, op_dat ux, op_dat uy);

void cub_div_weak(DGMesh *mesh, op_dat u, op_dat v, op_dat res);

void inv_mass(DGMesh *mesh, op_dat u);

#endif
