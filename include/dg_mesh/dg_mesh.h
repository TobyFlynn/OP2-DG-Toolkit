#ifndef __DG_MESH_H
#define __DG_MESH_H

#include "dg_compiler_defs.h"

#include "op_seq.h"

#include <vector>

class DGMesh {
public:
  virtual void init() = 0;

  virtual void update_order(int new_order, std::vector<op_dat> &dats_to_interp) = 0;
  virtual void update_order_sp(int new_order, std::vector<op_dat> &dats_to_interp) = 0;

  int order_int;

  op_set cells, faces, bfaces;
  op_map face2cells, bface2cells;
  op_dat order, J, geof;
};

#endif
