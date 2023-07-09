#ifndef __DG_MESH_2D_H
#define __DG_MESH_2D_H

#include "dg_compiler_defs.h"

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
  DGMesh2D(std::string &meshFile, bool overInt = true);
  ~DGMesh2D();
  void init() override;
  void update_order(op_dat new_orders, std::vector<op_dat> &dats_to_interp);
  void update_order(int new_order, std::vector<op_dat> &dats_to_interp) override;
  void interp_to_max_order(std::vector<op_dat> &dats_in, std::vector<op_dat> &dats_out);

  int get_local_vec_unknowns();

  // Operators
  void div(op_dat u, op_dat v, op_dat res);
  void div_with_central_flux(op_dat u, op_dat v, op_dat res);
  void div_with_central_flux_over_int(op_dat u, op_dat v, op_dat res);
  void div_weak(op_dat u, op_dat v, op_dat res);
  void curl(op_dat u, op_dat v, op_dat res);
  void grad(op_dat u, op_dat ux, op_dat uy);
  void grad_with_central_flux(op_dat u, op_dat ux, op_dat uy);
  void grad_with_central_flux_over_int(op_dat u, op_dat ux, op_dat uy);
  void cub_grad(op_dat u, op_dat ux, op_dat uy);
  void cub_grad_with_central_flux(op_dat u, op_dat ux, op_dat uy);
  void cub_div(op_dat u, op_dat v, op_dat res);
  void cub_div_with_central_flux_no_inv_mass(op_dat u, op_dat v, op_dat res);
  void cub_div_with_central_flux(op_dat u, op_dat v, op_dat res);
  void cub_grad_weak(op_dat u, op_dat ux, op_dat uy);
  void cub_grad_weak_with_central_flux(op_dat u, op_dat ux, op_dat uy);
  void cub_div_weak(op_dat u, op_dat v, op_dat res);
  void cub_div_weak_with_central_flux(op_dat u, op_dat v, op_dat res);
  void mass(op_dat u);
  void inv_mass(op_dat u);
  void avg(op_dat in, op_dat out);
  void jump(op_dat in, op_dat out);
  void interp_dat_between_orders(int old_order, int new_order, op_dat in, op_dat out);
  void interp_dat_between_orders(int old_order, int new_order, op_dat in);

  // OP2 stuff
  op_dat node_coords, nodeX, nodeY, x, y, rx, ry, sx, sy;
  op_dat nx, ny, sJ, fscale, bnx, bny, bsJ, bfscale;
  op_dat bedge_type, edgeNum, bedgeNum, reverse;
  op_dat nx_c, ny_c, sJ_c, fscale_c;
  op_dat op_tmp[4];

  DGCubatureData *cubature;
  DGGaussData *gauss;
private:
  void update_mesh_constants();

  bool over_integrate;
};

#endif
