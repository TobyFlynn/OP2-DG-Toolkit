#ifndef __DG_MESH_2D_H
#define __DG_MESH_2D_H

#include "dg_compiler_defs.h"

#include "dg_mesh.h"

#include "op_seq.h"

#include <string>
#include <vector>

class DGMesh2D : public DGMesh {
public:
  DGMesh2D(std::string &meshFile);
  ~DGMesh2D();
  void init() override;
  void update_order_sp(int new_order, std::vector<op_dat> &dats_to_interp) override;
  void update_order(int new_order, std::vector<op_dat> &dats_to_interp) override;

  // Operators
  void div(op_dat u, op_dat v, op_dat res);
  void div_with_central_flux(op_dat u, op_dat v, op_dat res);
  void div_weak(op_dat u, op_dat v, op_dat res);
  void curl(op_dat u, op_dat v, op_dat res);
  void grad(op_dat u, op_dat ux, op_dat uy);
  void grad_weak(op_dat u, op_dat ux, op_dat uy);
  void grad_with_central_flux(op_dat u, op_dat ux, op_dat uy);
  void grad_over_int_with_central_flux(op_dat u, op_dat ux, op_dat uy);
  void mass(op_dat u);
  void mass_sp(op_dat u);
  void inv_mass(op_dat u);
  void avg(op_dat in, op_dat out);
  void jump(op_dat in, op_dat out);
  void avg_sp(op_dat in, op_dat out);
  void jump_sp(op_dat in, op_dat out);
  void interp_dat_between_orders(int old_order, int new_order, op_dat in, op_dat out);
  void interp_dat_between_orders(int old_order, int new_order, op_dat in);
  void interp_dat_between_orders_sp(int old_order, int new_order, op_dat in);

  // OP2 stuff
  op_dat node_coords, nodeX, nodeY, x, y;
  op_dat nx, ny, sJ, fscale, bnx, bny, bsJ, bfscale;
  op_dat bedge_type, edgeNum, bedgeNum, reverse;
  op_dat nx_c, ny_c, sJ_c, fscale_c;

private:
  void calc_mesh_constants();
};

#endif
