#include "dg_mesh.h"

#include "op_seq.h"

#include <string>
#include <memory>

#include "constants/all_constants.h"
#include "dg_blas_calls.h"

using namespace std;

void set_cuda_const();

DGCubatureData::DGCubatureData(DGMesh *m) {
  mesh = m;

  rx_data  = (double *)calloc(46 * mesh->numCells, sizeof(double));
  sx_data  = (double *)calloc(46 * mesh->numCells, sizeof(double));
  ry_data  = (double *)calloc(46 * mesh->numCells, sizeof(double));
  sy_data  = (double *)calloc(46 * mesh->numCells, sizeof(double));
  J_data   = (double *)calloc(46 * mesh->numCells, sizeof(double));
  mm_data  = (double *)calloc(15 * 15 * mesh->numCells, sizeof(double));
  tmp_data = (double *)calloc(46 * 15 * mesh->numCells, sizeof(double));

  for(int i = 0; i < 4; i++) {
    op_tmp_data[i] = (double *)calloc(46 * mesh->numCells, sizeof(double));
  }

  rx  = op_decl_dat(mesh->cells, 46, "double", rx_data, "cub-rx");
  sx  = op_decl_dat(mesh->cells, 46, "double", sx_data, "cub-sx");
  ry  = op_decl_dat(mesh->cells, 46, "double", ry_data, "cub-ry");
  sy  = op_decl_dat(mesh->cells, 46, "double", sy_data, "cub-sy");
  J   = op_decl_dat(mesh->cells, 46, "double", J_data, "cub-J");
  mm  = op_decl_dat(mesh->cells, 15 * 15, "double", mm_data, "cub-mm");
  tmp = op_decl_dat(mesh->cells, 46 * 15, "double", tmp_data, "cub-tmp");

  for(int i = 0; i < 4; i++) {
    string tmpname = "cub-op_tmp" + to_string(i);
    op_tmp[i] = op_decl_dat(mesh->cells, 46, "double", op_tmp_data[i], tmpname.c_str());
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
  // Calculate geometric factors for cubature volume nodes
  op2_gemv(true, 46, 15, 1.0, constants->get_ptr(DGConstants::CUB_DR), 15, mesh->x, 0.0, rx);
  op2_gemv(true, 46, 15, 1.0, constants->get_ptr(DGConstants::CUB_DS), 15, mesh->x, 0.0, sx);
  op2_gemv(true, 46, 15, 1.0, constants->get_ptr(DGConstants::CUB_DR), 15, mesh->y, 0.0, ry);
  op2_gemv(true, 46, 15, 1.0, constants->get_ptr(DGConstants::CUB_DS), 15, mesh->y, 0.0, sy);

  op_par_loop(init_cubature, "init_cubature", mesh->cells,
              op_arg_dat(rx,   -1, OP_ID, 46, "double", OP_RW),
              op_arg_dat(sx,   -1, OP_ID, 46, "double", OP_RW),
              op_arg_dat(ry,   -1, OP_ID, 46, "double", OP_RW),
              op_arg_dat(sy,   -1, OP_ID, 46, "double", OP_RW),
              op_arg_dat(J,    -1, OP_ID, 46, "double", OP_WRITE),
              op_arg_dat(tmp,  -1, OP_ID, 46 * 15, "double", OP_WRITE));
  // Temp is in row-major at this point
  op2_gemm(false, true, 15, 15, 46, 1.0, constants->get_ptr(DGConstants::CUB_V), 15, tmp, 15, 0.0, mm, 15);
  // mm is in col-major at this point
}

DGGaussData::DGGaussData(DGMesh *m) {
  mesh = m;

  x_data  = (double *)calloc(21 * mesh->numCells, sizeof(double));
  y_data  = (double *)calloc(21 * mesh->numCells, sizeof(double));
  rx_data = (double *)calloc(21 * mesh->numCells, sizeof(double));
  sx_data = (double *)calloc(21 * mesh->numCells, sizeof(double));
  ry_data = (double *)calloc(21 * mesh->numCells, sizeof(double));
  sy_data = (double *)calloc(21 * mesh->numCells, sizeof(double));
  sJ_data = (double *)calloc(21 * mesh->numCells, sizeof(double));
  nx_data = (double *)calloc(21 * mesh->numCells, sizeof(double));
  ny_data = (double *)calloc(21 * mesh->numCells, sizeof(double));

  x  = op_decl_dat(mesh->cells, 21, "double", x_data, "gauss-x");
  y  = op_decl_dat(mesh->cells, 21, "double", y_data, "gauss-y");
  rx = op_decl_dat(mesh->cells, 21, "double", rx_data, "gauss-rx");
  sx = op_decl_dat(mesh->cells, 21, "double", sx_data, "gauss-sx");
  ry = op_decl_dat(mesh->cells, 21, "double", ry_data, "gauss-ry");
  sy = op_decl_dat(mesh->cells, 21, "double", sy_data, "gauss-sy");
  sJ = op_decl_dat(mesh->cells, 21, "double", sJ_data, "gauss-sJ");
  nx = op_decl_dat(mesh->cells, 21, "double", nx_data, "gauss-nx");
  ny = op_decl_dat(mesh->cells, 21, "double", ny_data, "gauss-ny");
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
}

void DGGaussData::init() {
  op2_gemv(true, 21, 15, 1.0, constants->get_ptr(DGConstants::GAUSS_INTERP), 15, mesh->x, 0.0, x);
  op2_gemv(true, 21, 15, 1.0, constants->get_ptr(DGConstants::GAUSS_INTERP), 15, mesh->y, 0.0, y);

  // Initialise geometric factors for Gauss nodes
  init_gauss_blas(mesh, this);

  op_par_loop(init_gauss, "init_gauss", mesh->cells,
              op_arg_dat(rx, -1, OP_ID, 21, "double", OP_RW),
              op_arg_dat(sx, -1, OP_ID, 21, "double", OP_RW),
              op_arg_dat(ry, -1, OP_ID, 21, "double", OP_RW),
              op_arg_dat(sy, -1, OP_ID, 21, "double", OP_RW),
              op_arg_dat(nx, -1, OP_ID, 21, "double", OP_WRITE),
              op_arg_dat(ny, -1, OP_ID, 21, "double", OP_WRITE),
              op_arg_dat(sJ, -1, OP_ID, 21, "double", OP_WRITE));
}

DGMesh::DGMesh(double *coords_a, int *cells_a, int *edge2node_a,
               int *edge2cell_a, int *bedge2node_a, int *bedge2cell_a,
               int *bedge_type_a, int *edgeNum_a, int *bedgeNum_a,
               int numNodes_g_a, int numCells_g_a, int numEdges_g_a,
               int numBoundaryEdges_g_a, int numNodes_a, int numCells_a,
               int numEdges_a, int numBoundaryEdges_a) {
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
  x_data       = (double *)calloc(15 * numCells, sizeof(double));
  y_data       = (double *)calloc(15 * numCells, sizeof(double));
  rx_data      = (double *)calloc(15 * numCells, sizeof(double));
  ry_data      = (double *)calloc(15 * numCells, sizeof(double));
  sx_data      = (double *)calloc(15 * numCells, sizeof(double));
  sy_data      = (double *)calloc(15 * numCells, sizeof(double));
  nx_data      = (double *)calloc(15 * numCells, sizeof(double));
  ny_data      = (double *)calloc(15 * numCells, sizeof(double));
  J_data       = (double *)calloc(15 * numCells, sizeof(double));
  sJ_data      = (double *)calloc(15 * numCells, sizeof(double));
  fscale_data  = (double *)calloc(15 * numCells, sizeof(double));
  reverse_data = (bool *)calloc(numEdges, sizeof(bool));
  for(int i = 0; i < 4; i++) {
    op_tmp_data[i] = (double *)calloc(15 * numCells, sizeof(double));
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
  x = op_decl_dat(cells, 15, "double", x_data, "x");
  y = op_decl_dat(cells, 15, "double", y_data, "y");
    // Geometric factors that relate to mapping between global and local (cell) coordinates
  rx = op_decl_dat(cells, 15, "double", rx_data, "rx");
  ry = op_decl_dat(cells, 15, "double", ry_data, "ry");
  sx = op_decl_dat(cells, 15, "double", sx_data, "sx");
  sy = op_decl_dat(cells, 15, "double", sy_data, "sy");
    // Normals for each cell (calculated for each node on each edge, nodes can appear on multiple edges)
  nx = op_decl_dat(cells, 15, "double", nx_data, "nx");
  ny = op_decl_dat(cells, 15, "double", ny_data, "ny");
    // surface Jacobian / Jacobian (used when lifting the boundary fluxes)
  J          = op_decl_dat(cells, 15, "double", J_data, "J");
  sJ         = op_decl_dat(cells, 15, "double", sJ_data, "sJ");
  fscale     = op_decl_dat(cells, 15, "double", fscale_data, "fscale");
  bedge_type = op_decl_dat(bedges, 1, "int", bedge_type_data, "bedge_type");
  edgeNum    = op_decl_dat(edges, 2, "int", edgeNum_data, "edgeNum");
  bedgeNum   = op_decl_dat(bedges, 1, "int", bedgeNum_data, "bedgeNum");
  reverse    = op_decl_dat(edges, 1, "bool", reverse_data, "reverse");
  for(int i = 0; i < 4; i++) {
    string tmpname = "op_tmp" + to_string(i);
    op_tmp[i] = op_decl_dat(cells, 15, "double", op_tmp_data[i], tmpname.c_str());
  }

  #ifdef OP2_DG_CUDA
  set_cuda_const();
  #else
  op_decl_const(1, "double", &gam);
  op_decl_const(1, "double", &mu);
  op_decl_const(1, "double", &nu0);
  op_decl_const(1, "double", &nu1);
  op_decl_const(1, "double", &rho0);
  op_decl_const(1, "double", &rho1);
  op_decl_const(1, "double", &ren);
  op_decl_const(1, "double", &bc_mach);
  op_decl_const(1, "double", &bc_alpha);
  op_decl_const(1, "double", &bc_p);
  op_decl_const(1, "double", &bc_u);
  op_decl_const(1, "double", &bc_v);
  op_decl_const(15, "int", FMASK);
  op_decl_const(1, "double", &ic_u);
  op_decl_const(1, "double", &ic_v);
  op_decl_const(46, "double", cubW_g);
  op_decl_const(46*15, "double", cubV_g);
  op_decl_const(46*15, "double", cubVDr_g);
  op_decl_const(46*15, "double", cubVDs_g);
  op_decl_const(7*15, "double", gF0Dr_g);
  op_decl_const(7*15, "double", gF0Ds_g);
  op_decl_const(7*15, "double", gF1Dr_g);
  op_decl_const(7*15, "double", gF1Ds_g);
  op_decl_const(7*15, "double", gF2Dr_g);
  op_decl_const(7*15, "double", gF2Ds_g);
  op_decl_const(7, "double", gaussW_g);
  op_decl_const(7*15, "double", gFInterp0_g);
  op_decl_const(7*15, "double", gFInterp1_g);
  op_decl_const(7*15, "double", gFInterp2_g);
  op_decl_const(7*15, "double", gF0DrR_g);
  op_decl_const(7*15, "double", gF0DsR_g);
  op_decl_const(7*15, "double", gF1DrR_g);
  op_decl_const(7*15, "double", gF1DsR_g);
  op_decl_const(7*15, "double", gF2DrR_g);
  op_decl_const(7*15, "double", gF2DsR_g);
  op_decl_const(7*15, "double", gFInterp0R_g);
  op_decl_const(7*15, "double", gFInterp1R_g);
  op_decl_const(7*15, "double", gFInterp2R_g);
  op_decl_const(5, "double", lift_drag_vec);
  #endif

  cubature = new DGCubatureData(this);
  gauss = new DGGaussData(this);
}

DGMesh::~DGMesh() {
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
}

void DGMesh::init() {
  op_par_loop(init_nodes, "init_nodes", cells,
              op_arg_dat(node_coords, -3, cell2nodes, 2, "double", OP_READ),
              op_arg_dat(nodeX, -1, OP_ID, 3, "double", OP_WRITE),
              op_arg_dat(nodeY, -1, OP_ID, 3, "double", OP_WRITE));

  // Calculate geometric factors
  init_grid_blas(this);

  op_par_loop(init_grid, "init_grid", cells,
              op_arg_dat(rx, -1, OP_ID, 15, "double", OP_RW),
              op_arg_dat(ry, -1, OP_ID, 15, "double", OP_RW),
              op_arg_dat(sx, -1, OP_ID, 15, "double", OP_RW),
              op_arg_dat(sy, -1, OP_ID, 15, "double", OP_RW),
              op_arg_dat(nx, -1, OP_ID, 15, "double", OP_WRITE),
              op_arg_dat(ny, -1, OP_ID, 15, "double", OP_WRITE),
              op_arg_dat(J,  -1, OP_ID, 15, "double", OP_WRITE),
              op_arg_dat(sJ, -1, OP_ID, 15, "double", OP_WRITE),
              op_arg_dat(fscale, -1, OP_ID, 15, "double", OP_WRITE));

  op_par_loop(init_edges, "init_edges", edges,
              op_arg_dat(edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(nodeX, -2, edge2cells, 3, "double", OP_READ),
              op_arg_dat(nodeY, -2, edge2cells, 3, "double", OP_READ),
              op_arg_dat(reverse, -1, OP_ID, 1, "bool", OP_WRITE));

  cubature->init();
  gauss->init();
}
