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

  op_set nodes, cells, faces, bfaces;
  op_map cell2nodes, face2nodes, face2cells, bface2nodes, bface2cells;
  op_dat order, J, geof;
};

#endif
