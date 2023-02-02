#ifndef __DG_OPERATORS_H
#define __DG_OPERATORS_H

#include "op_seq.h"
#include "dg_mesh/dg_mesh_2d.h"

void div(DGMesh2D *mesh, op_dat u, op_dat v, op_dat res);
void div_with_central_flux(DGMesh2D *mesh, op_dat u, op_dat v, op_dat res);
void div_weak(DGMesh2D *mesh, op_dat u, op_dat v, op_dat res);

void curl(DGMesh2D *mesh, op_dat u, op_dat v, op_dat res);

void grad(DGMesh2D *mesh, op_dat u, op_dat ux, op_dat uy);
void grad_with_central_flux(DGMesh2D *mesh, op_dat u, op_dat ux, op_dat uy);

void cub_grad(DGMesh2D *mesh, op_dat u, op_dat ux, op_dat uy);
void cub_grad_with_central_flux(DGMesh2D *mesh, op_dat u, op_dat ux, op_dat uy);

void cub_div(DGMesh2D *mesh, op_dat u, op_dat v, op_dat res);
void cub_div_with_central_flux_no_inv_mass(DGMesh2D *mesh, op_dat u, op_dat v, op_dat res);
void cub_div_with_central_flux(DGMesh2D *mesh, op_dat u, op_dat v, op_dat res);

void cub_grad_weak(DGMesh2D *mesh, op_dat u, op_dat ux, op_dat uy);
void cub_grad_weak_with_central_flux(DGMesh2D *mesh, op_dat u, op_dat ux, op_dat uy);

void cub_div_weak(DGMesh2D *mesh, op_dat u, op_dat v, op_dat res);
void cub_div_weak_with_central_flux(DGMesh2D *mesh, op_dat u, op_dat v, op_dat res);

void inv_mass(DGMesh2D *mesh, op_dat u);

#endif
