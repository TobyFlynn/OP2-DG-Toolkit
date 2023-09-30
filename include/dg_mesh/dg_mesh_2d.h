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
  void update_order(op_dat new_orders, std::vector<op_dat> &dats_to_interp);
  void update_order(int new_order, std::vector<op_dat> &dats_to_interp) override;
  void interp_to_max_order(std::vector<op_dat> &dats_in, std::vector<op_dat> &dats_out);

  int get_local_vec_unknowns();

  // Operators
  void div(op_dat u, op_dat v, op_dat res);
  void div_with_central_flux(op_dat u, op_dat v, op_dat res);
  void div_weak(op_dat u, op_dat v, op_dat res);
  void curl(op_dat u, op_dat v, op_dat res);
  void grad(op_dat u, op_dat ux, op_dat uy);
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
  op_dat nodeX, nodeY, x, y, rx, ry, sx, sy;
  op_dat nx, ny, sJ, fscale, bnx, bny, bsJ, bfscale;
  op_dat bedge_type, edgeNum, bedgeNum, reverse;
  op_dat nx_c, ny_c, sJ_c, fscale_c;
  op_dat nx_c_new, ny_c_new, sJ_c_new, fscale_c_new;
  op_dat order, J, op_tmp[4];

private:
  void update_mesh_constants();
};

#endif
