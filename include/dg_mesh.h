#ifndef __DG_MESH_H
#define __DG_MESH_H

#include "op_seq.h"

#include <string>

#include "dg_global_constants.h"

class DGMesh {
public:
  DGMesh(double *coords_a, int *cells_a, int *edge2node_a, int *edge2cell_a,
         int *bedge2node_a, int *bedge2cell_a, int *bedge_type_a,
         int *edgeNum_a, int *bedgeNum_a, int numNodes_g_a, int numCells_g_a,
         int numEdges_g_a, int numBoundaryEdges_g_a, int numNodes_a,
         int numCells_a, int numEdges_a, int numBoundaryEdges_a);
  ~DGMesh();
  void init();
  // Pointers used when loading data
  double *coords_data;
  int *cells_data;
  int *edge2node_data;
  int *edge2cell_data;
  int *bedge2node_data;
  int *bedge2cell_data;
  int *bedge_type_data;
  int *edgeNum_data;
  int *bedgeNum_data;
  int numNodes_g, numCells_g, numEdges_g, numBoundaryEdges_g;
  int numNodes, numCells, numEdges, numBoundaryEdges;
  // OP2 stuff
  op_set nodes, cells, edges, bedges;
  op_map cell2nodes, edge2nodes, edge2cells, bedge2nodes, bedge2cells;
  op_dat node_coords, nodeX, nodeY, x, y, rx, ry, sx, sy, nx,
         ny, J, sJ, fscale, bedge_type, edgeNum, bedgeNum, reverse;
  op_dat op_tmp[4];
private:
  // Pointers to private memory
  double *nodeX_data;
  double *nodeY_data;
  double *x_data;
  double *y_data;
  double *rx_data;
  double *ry_data;
  double *sx_data;
  double *sy_data;
  double *nx_data;
  double *ny_data;
  double *J_data;
  double *sJ_data;
  double *fscale_data;
  bool *reverse_data;
  double *op_tmp_data[4];
};

#endif
