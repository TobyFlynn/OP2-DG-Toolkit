#ifndef __DG_DAT_POOL_H
#define __DG_DAT_POOL_H

#include "op_seq.h"

#include <vector>

#include "dg_mesh/dg_mesh_3d.h"

struct DGTempDat {
  op_dat dat;
  int ind;
};

class DGDatPool3D {
public:
  DGDatPool3D(DGMesh3D *m);
  ~DGDatPool3D();

  DGTempDat requestTempDatCells(const int dim);
  void releaseTempDatCells(DGTempDat tempDat);

  DGTempDat requestTempDatFaces(const int dim);
  void releaseTempDatFaces(DGTempDat tempDat);

  void report();
private:
  struct DatWrapper {
    op_dat dat;
    bool in_use;
  };

  DGMesh3D *mesh;

  std::vector<DatWrapper> cell_dats;
  std::vector<DatWrapper> face_dats;
};

#endif
