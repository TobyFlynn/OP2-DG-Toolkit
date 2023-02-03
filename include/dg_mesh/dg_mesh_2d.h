#ifndef __DG_MESH_2D_H
#define __DG_MESH_2D_H

#include "dg_mesh.h"

#include "op_seq.h"

#include <string>
#include <vector>

class DGMesh2D;

class DGCubatureData {
public:
  DGCubatureData(DGMesh2D *m);
  void init();
  void update_mesh_constants();

  // mm is stored in column major format
  op_dat rx, sx, ry, sy, J, mm;
  op_dat op_tmp[4], tmp;

private:
  DGMesh2D *mesh;
};

class DGGaussData {
public:
  DGGaussData(DGMesh2D *m);
  void init();
  void update_mesh_constants();

  op_dat x, y;
  op_dat rx, sx, ry, sy, sJ, nx, ny;
  op_dat op_tmp[3];
private:
  DGMesh2D *mesh;
};

class DGMesh2D : public DGMesh {
public:
  DGMesh2D(std::string &meshFile);
  ~DGMesh2D();
  void init() override;
  void update_order(op_dat new_orders, std::vector<op_dat> &dats_to_interpolate);
  void update_order(int new_order, std::vector<op_dat> &dats_to_interpolate);
  void interp_to_max_order(std::vector<op_dat> &dats_in, std::vector<op_dat> &dats_out);

  int get_local_vec_unknowns();

  // Operators
  void div(op_dat u, op_dat v, op_dat res);
  void div_with_central_flux(op_dat u, op_dat v, op_dat res);
  void div_weak(op_dat u, op_dat v, op_dat res);
  void curl(op_dat u, op_dat v, op_dat res);
  void grad(op_dat u, op_dat ux, op_dat uy);
  void grad_with_central_flux(op_dat u, op_dat ux, op_dat uy);
  void cub_grad(op_dat u, op_dat ux, op_dat uy);
  void cub_grad_with_central_flux(op_dat u, op_dat ux, op_dat uy);
  void cub_div(op_dat u, op_dat v, op_dat res);
  void cub_div_with_central_flux_no_inv_mass(op_dat u, op_dat v, op_dat res);
  void cub_div_with_central_flux(op_dat u, op_dat v, op_dat res);
  void cub_grad_weak(op_dat u, op_dat ux, op_dat uy);
  void cub_grad_weak_with_central_flux(op_dat u, op_dat ux, op_dat uy);
  void cub_div_weak(op_dat u, op_dat v, op_dat res);
  void cub_div_weak_with_central_flux(op_dat u, op_dat v, op_dat res);
  void inv_mass(op_dat u);

  // OP2 stuff
  op_dat node_coords, nodeX, nodeY, x, y, rx, ry, sx, sy, nx,
         ny, sJ, fscale, bedge_type, edgeNum, bedgeNum, reverse, order;
  op_dat op_tmp[4];

  DGCubatureData *cubature;
  DGGaussData *gauss;
private:
  void update_mesh_constants();
};

#endif
