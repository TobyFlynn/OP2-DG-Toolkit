#include "dg_mesh.h"

#include "op_seq.h"

#include <string>
#include <memory>
#include <iostream>

#include "dg_blas_calls.h"
#include "dg_compiler_defs.h"
#include "dg_constants.h"
#include "dg_op2_blas.h"
#include "dg_global_constants.h"

DGConstants *constants[DG_ORDER + 1];

using namespace std;

#ifdef OP2_DG_CUDA
void set_cuda_constants_OP2_DG_CUDA();
#endif

DGCubatureData::DGCubatureData(DGMesh *m) {
  mesh = m;

  rx_data    = (double *)calloc(DG_CUB_NP * mesh->numCells, sizeof(double));
  sx_data    = (double *)calloc(DG_CUB_NP * mesh->numCells, sizeof(double));
  ry_data    = (double *)calloc(DG_CUB_NP * mesh->numCells, sizeof(double));
  sy_data    = (double *)calloc(DG_CUB_NP * mesh->numCells, sizeof(double));
  J_data     = (double *)calloc(DG_CUB_NP * mesh->numCells, sizeof(double));
  mm_data    = (double *)calloc(DG_NP * DG_NP * mesh->numCells, sizeof(double));
  tmp_data   = (double *)calloc(DG_CUB_NP * DG_NP * mesh->numCells, sizeof(double));

  for(int i = 0; i < 4; i++) {
    op_tmp_data[i] = (double *)calloc(DG_CUB_NP * mesh->numCells, sizeof(double));
  }

  rx    = op_decl_dat(mesh->cells, DG_CUB_NP, "double", rx_data, "cub-rx");
  sx    = op_decl_dat(mesh->cells, DG_CUB_NP, "double", sx_data, "cub-sx");
  ry    = op_decl_dat(mesh->cells, DG_CUB_NP, "double", ry_data, "cub-ry");
  sy    = op_decl_dat(mesh->cells, DG_CUB_NP, "double", sy_data, "cub-sy");
  J     = op_decl_dat(mesh->cells, DG_CUB_NP, "double", J_data, "cub-J");
  mm    = op_decl_dat(mesh->cells, DG_NP * DG_NP, "double", mm_data, "cub-mm");
  tmp   = op_decl_dat(mesh->cells, DG_CUB_NP * DG_NP, "double", tmp_data, "cub-tmp");

  for(int i = 0; i < 4; i++) {
    string tmpname = "cub-op_tmp" + to_string(i);
    op_tmp[i] = op_decl_dat(mesh->cells, DG_CUB_NP, "double", op_tmp_data[i], tmpname.c_str());
  }
}

DGCubatureData::~DGCubatureData() {
  free(rx_data);
  free(sx_data);
  free(ry_data);
  free(sy_data);
  free(J_data);
  free(mm_data);
  free(tmp_data);

  for(int i = 0; i < 4; i++) {
    free(op_tmp_data[i]);
  }
}

void DGCubatureData::init() {
  update_mesh_constants();
}

void DGCubatureData::update_mesh_constants() {
  op2_gemv(mesh, false, 1.0, DGConstants::CUB_DR, mesh->x, 0.0, rx);
  op2_gemv(mesh, false, 1.0, DGConstants::CUB_DS, mesh->x, 0.0, sx);
  op2_gemv(mesh, false, 1.0, DGConstants::CUB_DR, mesh->y, 0.0, ry);
  op2_gemv(mesh, false, 1.0, DGConstants::CUB_DS, mesh->y, 0.0, sy);

  op_par_loop(init_cubature, "init_cubature", mesh->cells,
              op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(cubV_g, DG_ORDER * DG_CUB_NP * DG_NP, "double", OP_READ),
              op_arg_dat(rx,    -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(sx,    -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(ry,    -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(sy,    -1, OP_ID, DG_CUB_NP, "double", OP_RW),
              op_arg_dat(J,     -1, OP_ID, DG_CUB_NP, "double", OP_WRITE),
              op_arg_dat(tmp,   -1, OP_ID, DG_CUB_NP * DG_NP, "double", OP_WRITE));

  op_par_loop(cub_mm_init, "cub_mm_init", mesh->cells,
              op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(cubV_g, DG_ORDER * DG_CUB_NP * DG_NP, "double", OP_READ),
              op_arg_dat(tmp,   -1, OP_ID, DG_CUB_NP * DG_NP, "double", OP_READ),
              op_arg_dat(mm,    -1, OP_ID, DG_NP * DG_NP, "double", OP_WRITE));
}

DGGaussData::DGGaussData(DGMesh *m) {
  mesh = m;

  x_data  = (double *)calloc(DG_G_NP * mesh->numCells, sizeof(double));
  y_data  = (double *)calloc(DG_G_NP * mesh->numCells, sizeof(double));
  rx_data = (double *)calloc(DG_G_NP * mesh->numCells, sizeof(double));
  sx_data = (double *)calloc(DG_G_NP * mesh->numCells, sizeof(double));
  ry_data = (double *)calloc(DG_G_NP * mesh->numCells, sizeof(double));
  sy_data = (double *)calloc(DG_G_NP * mesh->numCells, sizeof(double));
  sJ_data = (double *)calloc(DG_G_NP * mesh->numCells, sizeof(double));
  nx_data = (double *)calloc(DG_G_NP * mesh->numCells, sizeof(double));
  ny_data = (double *)calloc(DG_G_NP * mesh->numCells, sizeof(double));

  for(int i = 0; i < 3; i++) {
    op_tmp_data[i] = (double *)calloc(DG_G_NP * mesh->numCells, sizeof(double));
  }

  x  = op_decl_dat(mesh->cells, DG_G_NP, "double", x_data, "gauss-x");
  y  = op_decl_dat(mesh->cells, DG_G_NP, "double", y_data, "gauss-y");
  rx = op_decl_dat(mesh->cells, DG_G_NP, "double", rx_data, "gauss-rx");
  sx = op_decl_dat(mesh->cells, DG_G_NP, "double", sx_data, "gauss-sx");
  ry = op_decl_dat(mesh->cells, DG_G_NP, "double", ry_data, "gauss-ry");
  sy = op_decl_dat(mesh->cells, DG_G_NP, "double", sy_data, "gauss-sy");
  sJ = op_decl_dat(mesh->cells, DG_G_NP, "double", sJ_data, "gauss-sJ");
  nx = op_decl_dat(mesh->cells, DG_G_NP, "double", nx_data, "gauss-nx");
  ny = op_decl_dat(mesh->cells, DG_G_NP, "double", ny_data, "gauss-ny");

  for(int i = 0; i < 3; i++) {
    string tmpname = "gauss-op_tmp" + to_string(i);
    op_tmp[i] = op_decl_dat(mesh->cells, DG_G_NP, "double", op_tmp_data[i], tmpname.c_str());
  }
}

DGGaussData::~DGGaussData() {
  free(x_data);
  free(y_data);
  free(rx_data);
  free(sx_data);
  free(ry_data);
  free(sy_data);
  free(sJ_data);
  free(nx_data);
  free(ny_data);
  for(int i = 0; i < 3; i++) {
    free(op_tmp_data[i]);
  }
}

void DGGaussData::init() {
  update_mesh_constants();
}

void DGGaussData::update_mesh_constants() {
  op2_gemv(mesh, false, 1.0, DGConstants::GAUSS_INTERP, mesh->x, 0.0, x);
  op2_gemv(mesh, false, 1.0, DGConstants::GAUSS_INTERP, mesh->y, 0.0, y);

  op_par_loop(pre_init_gauss, "pre_init_gauss", mesh->cells,
              op_arg_dat(mesh->order,  -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(mesh->x, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(mesh->y, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_gbl(gF0Dr_g, DG_ORDER * DG_GF_NP * DG_NP, "double", OP_READ),
              op_arg_gbl(gF0Ds_g, DG_ORDER * DG_GF_NP * DG_NP, "double", OP_READ),
              op_arg_gbl(gF1Dr_g, DG_ORDER * DG_GF_NP * DG_NP, "double", OP_READ),
              op_arg_gbl(gF1Ds_g, DG_ORDER * DG_GF_NP * DG_NP, "double", OP_READ),
              op_arg_gbl(gF2Dr_g, DG_ORDER * DG_GF_NP * DG_NP, "double", OP_READ),
              op_arg_gbl(gF2Ds_g, DG_ORDER * DG_GF_NP * DG_NP, "double", OP_READ),
              op_arg_dat(rx, -1, OP_ID, DG_G_NP, "double", OP_WRITE),
              op_arg_dat(sx, -1, OP_ID, DG_G_NP, "double", OP_WRITE),
              op_arg_dat(ry, -1, OP_ID, DG_G_NP, "double", OP_WRITE),
              op_arg_dat(sy, -1, OP_ID, DG_G_NP, "double", OP_WRITE));

  op_par_loop(init_gauss, "init_gauss", mesh->cells,
              op_arg_dat(mesh->order,  -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(rx,     -1, OP_ID, DG_G_NP, "double", OP_RW),
              op_arg_dat(sx,     -1, OP_ID, DG_G_NP, "double", OP_RW),
              op_arg_dat(ry,     -1, OP_ID, DG_G_NP, "double", OP_RW),
              op_arg_dat(sy,     -1, OP_ID, DG_G_NP, "double", OP_RW),
              op_arg_dat(nx,     -1, OP_ID, DG_G_NP, "double", OP_WRITE),
              op_arg_dat(ny,     -1, OP_ID, DG_G_NP, "double", OP_WRITE),
              op_arg_dat(sJ,     -1, OP_ID, DG_G_NP, "double", OP_WRITE));
}

DGMesh::DGMesh(double *coords_a, int *cells_a, int *edge2node_a,
               int *edge2cell_a, int *bedge2node_a, int *bedge2cell_a,
               int *bedge_type_a, int *edgeNum_a, int *bedgeNum_a,
               int numNodes_g_a, int numCells_g_a, int numEdges_g_a,
               int numBoundaryEdges_g_a, int numNodes_a, int numCells_a,
               int numEdges_a, int numBoundaryEdges_a) {
  init_blas();
  // Calculate DG constants
  // Not currently considering 0th order
  constants[0] = nullptr;
  for(int p = 1; p <= DG_ORDER; p++) {
    constants[p] = new DGConstants(p);
  }
  // Now that all constants have been calculated,
  // calc interpolation matrices between different orders
  for(int p = 1; p <= DG_ORDER; p++) {
    constants[p]->calc_interp_mats();
  }

  coords_data        = coords_a;
  cells_data         = cells_a;
  edge2node_data     = edge2node_a;
  edge2cell_data     = edge2cell_a;
  bedge2node_data    = bedge2node_a;
  bedge2cell_data    = bedge2cell_a;
  bedge_type_data    = bedge_type_a;
  edgeNum_data       = edgeNum_a;
  bedgeNum_data      = bedgeNum_a;
  numNodes_g         = numNodes_g_a;
  numCells_g         = numCells_g_a;
  numEdges_g         = numEdges_g_a;
  numBoundaryEdges_g = numBoundaryEdges_g_a;
  numNodes           = numNodes_a;
  numCells           = numCells_a;
  numEdges           = numEdges_a;
  numBoundaryEdges   = numBoundaryEdges_a;

  // Initialise memory
  nodeX_data   = (double*)calloc(3 * numCells, sizeof(double));
  nodeY_data   = (double*)calloc(3 * numCells, sizeof(double));
  x_data       = (double *)calloc(DG_NP * numCells, sizeof(double));
  y_data       = (double *)calloc(DG_NP * numCells, sizeof(double));
  rx_data      = (double *)calloc(DG_NP * numCells, sizeof(double));
  ry_data      = (double *)calloc(DG_NP * numCells, sizeof(double));
  sx_data      = (double *)calloc(DG_NP * numCells, sizeof(double));
  sy_data      = (double *)calloc(DG_NP * numCells, sizeof(double));
  nx_data      = (double *)calloc(3 * DG_NPF * numCells, sizeof(double));
  ny_data      = (double *)calloc(3 * DG_NPF * numCells, sizeof(double));
  J_data       = (double *)calloc(DG_NP * numCells, sizeof(double));
  sJ_data      = (double *)calloc(3 * DG_NPF * numCells, sizeof(double));
  fscale_data  = (double *)calloc(3 * DG_NPF * numCells, sizeof(double));
  reverse_data = (bool *)calloc(numEdges, sizeof(bool));
  order_data   = (int *)calloc(numCells, sizeof(int));
  for(int i = 0; i < 4; i++) {
    op_tmp_data[i] = (double *)calloc(DG_NP * numCells, sizeof(double));
  }

  // Initialise OP2
  // Declare OP2 sets
  nodes  = op_decl_set(numNodes, "nodes");
  cells  = op_decl_set(numCells, "cells");
  edges  = op_decl_set(numEdges, "edges");
  bedges = op_decl_set(numBoundaryEdges, "bedges");

  // Declare OP2 maps
  cell2nodes  = op_decl_map(cells, nodes, 3, cells_data, "cell2nodes");
  edge2nodes  = op_decl_map(edges, nodes, 2, edge2node_data, "edge2nodes");
  edge2cells  = op_decl_map(edges, cells, 2, edge2cell_data, "edge2cells");
  bedge2nodes = op_decl_map(bedges, nodes, 2, bedge2node_data, "bedge2nodes");
  bedge2cells = op_decl_map(bedges, cells, 1, bedge2cell_data, "bedge2cells");

  // Declare OP2 datasets
  // Structure: {x, y}
  node_coords = op_decl_dat(nodes, 2, "double", coords_data, "node_coords");
  // Coords of nodes per cell
  nodeX = op_decl_dat(cells, 3, "double", nodeX_data, "nodeX");
  nodeY = op_decl_dat(cells, 3, "double", nodeY_data, "nodeY");
  // The x and y coordinates of all the solution points in a cell
  x = op_decl_dat(cells, DG_NP, "double", x_data, "x");
  y = op_decl_dat(cells, DG_NP, "double", y_data, "y");
  // Geometric factors that relate to mapping between global and local (cell) coordinates
  rx = op_decl_dat(cells, DG_NP, "double", rx_data, "rx");
  ry = op_decl_dat(cells, DG_NP, "double", ry_data, "ry");
  sx = op_decl_dat(cells, DG_NP, "double", sx_data, "sx");
  sy = op_decl_dat(cells, DG_NP, "double", sy_data, "sy");
  // Normals for each cell (calculated for each node on each edge, nodes can appear on multiple edges)
  nx = op_decl_dat(cells, 3 * DG_NPF, "double", nx_data, "nx");
  ny = op_decl_dat(cells, 3 * DG_NPF, "double", ny_data, "ny");
  // surface Jacobian / Jacobian (used when lifting the boundary fluxes)
  J          = op_decl_dat(cells, DG_NP, "double", J_data, "J");
  sJ         = op_decl_dat(cells, 3 * DG_NPF, "double", sJ_data, "sJ");
  fscale     = op_decl_dat(cells, 3 * DG_NPF, "double", fscale_data, "fscale");
  bedge_type = op_decl_dat(bedges, 1, "int", bedge_type_data, "bedge_type");
  edgeNum    = op_decl_dat(edges, 2, "int", edgeNum_data, "edgeNum");
  bedgeNum   = op_decl_dat(bedges, 1, "int", bedgeNum_data, "bedgeNum");
  reverse    = op_decl_dat(edges, 1, "bool", reverse_data, "reverse");
  order      = op_decl_dat(cells, 1, "int", order_data, "order");
  for(int i = 0; i < 4; i++) {
    string tmpname = "op_tmp" + to_string(i);
    op_tmp[i] = op_decl_dat(cells, DG_NP, "double", op_tmp_data[i], tmpname.c_str());
  }

  #ifdef OP2_DG_CUDA
  set_cuda_constants_OP2_DG_CUDA();
  #else
  op_decl_const(DG_ORDER * 5, "int", DG_CONSTANTS);
  op_decl_const(DG_ORDER * DG_NPF * 3, "int", FMASK);
  op_decl_const(DG_ORDER * DG_CUB_NP, "double", cubW_g);
  op_decl_const(DG_ORDER * DG_GF_NP, "double", gaussW_g);
  #endif
 
  cubature = new DGCubatureData(this);
  gauss = new DGGaussData(this);
}

DGMesh::~DGMesh() {
  for(int p = 1; p <= DG_ORDER; p++) {
    delete constants[p];
  }
  free(coords_data);
  free(cells_data);
  free(edge2node_data);
  free(edge2cell_data);
  free(bedge2node_data);
  free(bedge2cell_data);
  free(bedge_type_data);
  free(edgeNum_data);
  free(bedgeNum_data);

  free(nodeX_data);
  free(nodeY_data);
  free(x_data);
  free(y_data);
  free(rx_data);
  free(ry_data);
  free(sx_data);
  free(sy_data);
  free(nx_data);
  free(ny_data);
  free(J_data);
  free(sJ_data);
  free(fscale_data);
  free(reverse_data);
  for(int i = 0; i < 4; i++) {
    free(op_tmp_data[i]);
  }

  delete cubature;
  delete gauss;
  destroy_blas();
}

void DGMesh::init() {
  // Initialise the order to the max order to start with
  op_par_loop(init_order, "init_order", cells,
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_WRITE));

  op_par_loop(init_nodes, "init_nodes", cells,
              op_arg_dat(node_coords, -3, cell2nodes, 2, "double", OP_READ),
              op_arg_dat(nodeX, -1, OP_ID, 3, "double", OP_WRITE),
              op_arg_dat(nodeY, -1, OP_ID, 3, "double", OP_WRITE));

  // Calculate geometric factors
  op_par_loop(calc_geom, "calc_geom", cells,
              op_arg_dat(order,  -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(r_g, DG_ORDER * DG_NP, "double", OP_READ),
              op_arg_gbl(s_g, DG_ORDER * DG_NP, "double", OP_READ),
              op_arg_dat(nodeX, -1, OP_ID, 3, "double", OP_READ),
              op_arg_dat(nodeY, -1, OP_ID, 3, "double", OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_WRITE),
              op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_WRITE));
  
  op2_gemv(this, false, 1.0, DGConstants::DR, x, 0.0, rx);
  op2_gemv(this, false, 1.0, DGConstants::DS, x, 0.0, sx);
  op2_gemv(this, false, 1.0, DGConstants::DR, y, 0.0, ry);
  op2_gemv(this, false, 1.0, DGConstants::DS, y, 0.0, sy);

  op_par_loop(init_grid, "init_grid", cells,
              op_arg_dat(order,  -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(rx,     -1, OP_ID, DG_NP, "double", OP_RW),
              op_arg_dat(ry,     -1, OP_ID, DG_NP, "double", OP_RW),
              op_arg_dat(sx,     -1, OP_ID, DG_NP, "double", OP_RW),
              op_arg_dat(sy,     -1, OP_ID, DG_NP, "double", OP_RW),
              op_arg_dat(nx,     -1, OP_ID, 3 * DG_NPF, "double", OP_WRITE),
              op_arg_dat(ny,     -1, OP_ID, 3 * DG_NPF, "double", OP_WRITE),
              op_arg_dat(J,      -1, OP_ID, DG_NP, "double", OP_WRITE),
              op_arg_dat(sJ,     -1, OP_ID, 3 * DG_NPF, "double", OP_WRITE),
              op_arg_dat(fscale, -1, OP_ID, 3 * DG_NPF, "double", OP_WRITE));

  op_par_loop(init_edges, "init_edges", edges,
              op_arg_dat(edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(nodeX, -2, edge2cells, 3, "double", OP_READ),
              op_arg_dat(nodeY, -2, edge2cells, 3, "double", OP_READ),
              op_arg_dat(reverse, -1, OP_ID, 1, "bool", OP_WRITE));

  cubature->init();
  gauss->init();
}

void DGMesh::update_mesh_constants() {
  op_par_loop(calc_geom, "calc_geom", cells,
              op_arg_dat(order,  -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(r_g, DG_ORDER * DG_NP, "double", OP_READ),
              op_arg_gbl(s_g, DG_ORDER * DG_NP, "double", OP_READ),
              op_arg_dat(nodeX, -1, OP_ID, 3, "double", OP_READ),
              op_arg_dat(nodeY, -1, OP_ID, 3, "double", OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_WRITE),
              op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_WRITE));
  
  op2_gemv(this, false, 1.0, DGConstants::DR, x, 0.0, rx);
  op2_gemv(this, false, 1.0, DGConstants::DS, x, 0.0, sx);
  op2_gemv(this, false, 1.0, DGConstants::DR, y, 0.0, ry);
  op2_gemv(this, false, 1.0, DGConstants::DS, y, 0.0, sy);

  op_par_loop(init_grid, "init_grid", cells,
              op_arg_dat(order,  -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(rx,     -1, OP_ID, DG_NP, "double", OP_RW),
              op_arg_dat(ry,     -1, OP_ID, DG_NP, "double", OP_RW),
              op_arg_dat(sx,     -1, OP_ID, DG_NP, "double", OP_RW),
              op_arg_dat(sy,     -1, OP_ID, DG_NP, "double", OP_RW),
              op_arg_dat(nx,     -1, OP_ID, 3 * DG_NPF, "double", OP_WRITE),
              op_arg_dat(ny,     -1, OP_ID, 3 * DG_NPF, "double", OP_WRITE),
              op_arg_dat(J,      -1, OP_ID, DG_NP, "double", OP_WRITE),
              op_arg_dat(sJ,     -1, OP_ID, 3 * DG_NPF, "double", OP_WRITE),
              op_arg_dat(fscale, -1, OP_ID, 3 * DG_NPF, "double", OP_WRITE));

  cubature->update_mesh_constants();
  gauss->update_mesh_constants();
}

void DGMesh::update_order(op_dat new_orders,
                          std::vector<op_dat> &dats_to_interpolate) {
  // Interpolate dats first (assumes all these dats are of size DG_NP)
  for(int i = 0; i < dats_to_interpolate.size(); i++) {
    if(dats_to_interpolate[i]->dim != DG_NP) {
      std::cerr << "Interpolating between orders for non DG_NP dim dats is not implemented ...  exiting" << std::endl;
      exit(-1);
    }
    op_par_loop(interp_dat_to_new_order, "interp_dat_to_new_order", cells,
                op_arg_gbl(order_interp_g, DG_ORDER * DG_ORDER * DG_NP * DG_NP, "double", OP_READ),
                op_arg_dat(order,      -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(new_orders, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(dats_to_interpolate[i], -1, OP_ID, DG_NP, "double", OP_RW));
  }

  // Copy across new orders
  op_par_loop(copy_new_orders, "copy_new_orders", cells,
              op_arg_dat(new_orders,  -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(order,  -1, OP_ID, 1, "int", OP_WRITE));

  // Update mesh constants for new orders
  update_mesh_constants();
}

void DGMesh::update_order(int new_order,
                          std::vector<op_dat> &dats_to_interpolate) {
  // Interpolate dats first (assumes all these dats are of size DG_NP)
  for(int i = 0; i < dats_to_interpolate.size(); i++) {
    if(dats_to_interpolate[i]->dim != DG_NP) {
      std::cerr << "Interpolating between orders for non DG_NP dim dats is not implemented ...  exiting" << std::endl;
      exit(-1);
    }
    op_par_loop(interp_dat_to_new_order_int, "interp_dat_to_new_order_int", cells,
                op_arg_gbl(order_interp_g, DG_ORDER * DG_ORDER * DG_NP * DG_NP, "double", OP_READ),
                op_arg_dat(order,    -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&new_order, 1, "int", OP_READ),
                op_arg_dat(dats_to_interpolate[i], -1, OP_ID, DG_NP, "double", OP_RW));
  }

  // Copy across new orders
  op_par_loop(copy_new_orders_int, "copy_new_orders_int", cells,
              op_arg_gbl(&new_order, 1, "int", OP_READ),
              op_arg_dat(order,  -1, OP_ID, 1, "int", OP_WRITE));

  // Update mesh constants for new orders
  update_mesh_constants();
}

void DGMesh::interp_to_max_order(std::vector<op_dat> &dats_in,
                                 std::vector<op_dat> &dats_out) {
  if(dats_in.size() != dats_out.size()) {
    std::cerr << "Error must specify an output dat for each input when interpolating to max order ...  exiting" << std::endl;
    exit(-1);
  }

  // Interpolate each dat to order DG_ORDER
  for(int i = 0; i < dats_in.size(); i++) {
    if(dats_in[i]->dim != DG_NP) {
      std::cerr << "Interpolating between orders for non DG_NP dim dats is not implemented ...  exiting" << std::endl;
      exit(-1);
    }
    op_par_loop(interp_dat_to_max_order, "interp_dat_to_max_order", cells,
                op_arg_gbl(order_interp_g, DG_ORDER * DG_ORDER * DG_NP * DG_NP, "double", OP_READ),
                op_arg_dat(order,       -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(dats_in[i],  -1, OP_ID, DG_NP, "double", OP_READ),
                op_arg_dat(dats_out[i], -1, OP_ID, DG_NP, "double", OP_WRITE));
  }
}
