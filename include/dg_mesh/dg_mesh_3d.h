#ifndef __DG_MESH_3D_H
#define __DG_MESH_3D_H

#include "dg_compiler_defs.h"

#include "dg_mesh.h"

#include "op_seq.h"

#include <string>
#include <vector>

class DGMesh3D : public DGMesh {
public:
  DGMesh3D(std::string &meshFile);
  ~DGMesh3D();
  void init() override;
  void update_order(int new_order, std::vector<op_dat> &dats_to_interp) override;

  // Operators
  void grad(op_dat u, op_dat ux, op_dat uy, op_dat uz);
  void grad_with_central_flux(op_dat u, op_dat ux, op_dat uy, op_dat uz);
  void div(op_dat u, op_dat v, op_dat w, op_dat res);
  void div_with_central_flux(op_dat u, op_dat v, op_dat w, op_dat res);
  void div_weak(op_dat u, op_dat v, op_dat w, op_dat res);
  void div_weak_with_central_flux(op_dat u, op_dat v, op_dat w, op_dat res);
  void curl(op_dat u, op_dat v, op_dat w, op_dat resx, op_dat resy, op_dat resz);
  void mass(op_dat u);
  void inv_mass(op_dat u);

  int order_int;

  // OP2 stuff
  op_dat node_coords, nodeX, nodeY, nodeZ, x, y, z;
  op_dat rx, ry, rz, sx, sy, sz, tx, ty, tz;
  op_dat faceNum, bfaceNum, periodicFace, fmaskL, fmaskR, nx, ny, nz, sJ, fscale;
  op_dat bnx, bny, bnz, bsJ, bfscale;
  op_dat op_tmp[3], op_tmp_npf[3];
private:
  void calc_mesh_constants();
};

#endif
