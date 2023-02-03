#ifndef __DG_MESH_3D_H
#define __DG_MESH_3D_H

#include "dg_mesh.h"

#include "op_seq.h"

#include <string>
#include <vector>

class DGMesh3D : public DGMesh {
public:
  DGMesh3D(std::string &meshFile);
  ~DGMesh3D();
  void init() override;
  void update_order(int new_order, std::vector<op_dat> &dats_to_interp);

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