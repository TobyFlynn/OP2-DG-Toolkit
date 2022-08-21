#ifndef __DG_MESH_H
#define __DG_MESH_H

#include "op_seq.h"

#include <string>
#include <vector>

class DGMesh;

class DGCubatureData {
public:
  DGCubatureData(DGMesh *m);
  ~DGCubatureData();
  void init();
  void update_mesh_constants();

  // mm is stored in column major format
  op_dat rx, sx, ry, sy, J, mm;
  op_dat op_tmp[4], tmp;

private:
  DGMesh *mesh;

  double *rx_data, *sx_data, *ry_data, *sy_data, *J_data;
  double *mm_data, *op_tmp_data[4], *tmp_data;
};

class DGGaussData {
public:
  DGGaussData(DGMesh *m);
  ~DGGaussData();
  void init();
  void update_mesh_constants();

  op_dat x, y;
  op_dat rx, sx, ry, sy, sJ, nx, ny;
  op_dat op_tmp[3];
private:
  DGMesh *mesh;

  double *x_data, *y_data, *rx_data, *sx_data, *ry_data, *sy_data;
  double *sJ_data, *nx_data, *ny_data, *op_tmp_data[3];
};

class DGMesh {
public:
  DGMesh(double *coords_a, int *cells_a, int *edge2node_a, int *edge2cell_a,
         int *bedge2node_a, int *bedge2cell_a, int *bedge_type_a,
         int *edgeNum_a, int *bedgeNum_a, int numNodes_g_a, int numCells_g_a,
         int numEdges_g_a, int numBoundaryEdges_g_a, int numNodes_a,
         int numCells_a, int numEdges_a, int numBoundaryEdges_a);
  ~DGMesh();
  void init();
  void update_order(op_dat new_orders, std::vector<op_dat> &dats_to_interpolate);
  void update_order(int new_order, std::vector<op_dat> &dats_to_interpolate);
  void interp_to_max_order(std::vector<op_dat> &dats_in, std::vector<op_dat> &dats_out);

  int get_local_vec_unknowns();

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
         ny, J, sJ, fscale, bedge_type, edgeNum, bedgeNum, reverse, order;
  op_dat op_tmp[4];

  DGCubatureData *cubature;
  DGGaussData *gauss;
private:
  void update_mesh_constants();

  // Pointers to private memory
  double *nodeX_data, *nodeY_data, *x_data, *y_data;
  double *rx_data, *ry_data, *sx_data, *sy_data, *nx_data, *ny_data;
  double *J_data, *sJ_data, *fscale_data, *op_tmp_data[4];
  bool *reverse_data;
  int *order_data;
};

#endif
