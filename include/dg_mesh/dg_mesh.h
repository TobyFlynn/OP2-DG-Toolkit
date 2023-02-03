#ifndef __DG_MESH_H
#define __DG_MESH_H

#include "op_seq.h"

class DGMesh {
public:
  virtual void init() = 0;

  op_set nodes, cells, faces, bfaces;
  op_map cell2nodes, face2nodes, face2cells, bface2nodes, bface2cells;
  op_dat order, J;
};

#endif
