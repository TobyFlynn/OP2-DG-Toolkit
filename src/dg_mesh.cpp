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

  double *tmp_cub_np = (double *)calloc(DG_CUB_NP * mesh->cells->size, sizeof(double));
  double *tmp_np_np = (double *)calloc(DG_NP * DG_NP * mesh->cells->size, sizeof(double));
  double *tmp_cub_np_np = (double *)calloc(DG_CUB_NP * DG_NP * mesh->cells->size, sizeof(double));
  rx    = op_decl_dat(mesh->cells, DG_CUB_NP, "double", tmp_cub_np, "cub-rx");
  sx    = op_decl_dat(mesh->cells, DG_CUB_NP, "double", tmp_cub_np, "cub-sx");
  ry    = op_decl_dat(mesh->cells, DG_CUB_NP, "double", tmp_cub_np, "cub-ry");
  sy    = op_decl_dat(mesh->cells, DG_CUB_NP, "double", tmp_cub_np, "cub-sy");
  J     = op_decl_dat(mesh->cells, DG_CUB_NP, "double", tmp_cub_np, "cub-J");
  mm    = op_decl_dat(mesh->cells, DG_NP * DG_NP, "double", tmp_np_np, "cub-mm");
  tmp   = op_decl_dat(mesh->cells, DG_CUB_NP * DG_NP, "double", tmp_cub_np_np, "cub-tmp");

  for(int i = 0; i < 4; i++) {
    string tmpname = "cub-op_tmp" + to_string(i);
    op_tmp[i] = op_decl_dat(mesh->cells, DG_CUB_NP, "double", tmp_cub_np, tmpname.c_str());
  }
  free(tmp_cub_np_np);
  free(tmp_np_np);
  free(tmp_cub_np);
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

  double *tmp_g_np = (double *)calloc(DG_G_NP * mesh->cells->size, sizeof(double));
  x  = op_decl_dat(mesh->cells, DG_G_NP, "double", tmp_g_np, "gauss-x");
  y  = op_decl_dat(mesh->cells, DG_G_NP, "double", tmp_g_np, "gauss-y");
  rx = op_decl_dat(mesh->cells, DG_G_NP, "double", tmp_g_np, "gauss-rx");
  sx = op_decl_dat(mesh->cells, DG_G_NP, "double", tmp_g_np, "gauss-sx");
  ry = op_decl_dat(mesh->cells, DG_G_NP, "double", tmp_g_np, "gauss-ry");
  sy = op_decl_dat(mesh->cells, DG_G_NP, "double", tmp_g_np, "gauss-sy");
  sJ = op_decl_dat(mesh->cells, DG_G_NP, "double", tmp_g_np, "gauss-sJ");
  nx = op_decl_dat(mesh->cells, DG_G_NP, "double", tmp_g_np, "gauss-nx");
  ny = op_decl_dat(mesh->cells, DG_G_NP, "double", tmp_g_np, "gauss-ny");

  for(int i = 0; i < 3; i++) {
    string tmpname = "gauss-op_tmp" + to_string(i);
    op_tmp[i] = op_decl_dat(mesh->cells, DG_G_NP, "double", tmp_g_np, tmpname.c_str());
  }
  free(tmp_g_np);
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

DGMesh::DGMesh(std::string &meshFile) {
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

  // Initialise OP2
  // Declare OP2 sets
  nodes  = op_decl_set_hdf5(meshFile.c_str(), "nodes");
  cells  = op_decl_set_hdf5(meshFile.c_str(), "cells");
  edges  = op_decl_set_hdf5(meshFile.c_str(), "edges");
  bedges = op_decl_set_hdf5(meshFile.c_str(), "bedges");

  // Declare OP2 maps
  cell2nodes  = op_decl_map_hdf5(cells, nodes, 3, meshFile.c_str(), "cell2nodes");
  edge2nodes  = op_decl_map_hdf5(edges, nodes, 2, meshFile.c_str(), "edge2nodes");
  edge2cells  = op_decl_map_hdf5(edges, cells, 2, meshFile.c_str(), "edge2cells");
  bedge2nodes = op_decl_map_hdf5(bedges, nodes, 2, meshFile.c_str(), "bedge2nodes");
  bedge2cells = op_decl_map_hdf5(bedges, cells, 1, meshFile.c_str(), "bedge2cells");

  // Declare OP2 datasets from HDF5
  // Structure: {x, y}
  node_coords = op_decl_dat_hdf5(nodes, 2, "double", meshFile.c_str(), "node_coords");
  bedge_type  = op_decl_dat_hdf5(bedges, 1, "int", meshFile.c_str(), "bedge_type");
  edgeNum     = op_decl_dat_hdf5(edges, 2, "int", meshFile.c_str(), "edgeNum");
  bedgeNum    = op_decl_dat_hdf5(bedges, 1, "int", meshFile.c_str(), "bedgeNum");

  // Declare regular OP2 datasets
  // Coords of nodes per cell
  double *tmp_3_data = (double*)calloc(3 * cells->size, sizeof(double));
  nodeX = op_decl_dat(cells, 3, "double", tmp_3_data, "nodeX");
  nodeY = op_decl_dat(cells, 3, "double", tmp_3_data, "nodeY");
  free(tmp_3_data);

  double *tmp_np  = (double *)calloc(DG_NP * cells->size, sizeof(double));
  double *tmp_npf = (double *)calloc(3 * DG_NPF * cells->size, sizeof(double));
  // The x and y coordinates of all the solution points in a cell
  x = op_decl_dat(cells, DG_NP, "double", tmp_np, "x");
  y = op_decl_dat(cells, DG_NP, "double", tmp_np, "y");
  // Geometric factors that relate to mapping between global and local (cell) coordinates
  rx = op_decl_dat(cells, DG_NP, "double", tmp_np, "rx");
  ry = op_decl_dat(cells, DG_NP, "double", tmp_np, "ry");
  sx = op_decl_dat(cells, DG_NP, "double", tmp_np, "sx");
  sy = op_decl_dat(cells, DG_NP, "double", tmp_np, "sy");
  // Normals for each cell (calculated for each node on each edge, nodes can appear on multiple edges)
  nx = op_decl_dat(cells, 3 * DG_NPF, "double", tmp_npf, "nx");
  ny = op_decl_dat(cells, 3 * DG_NPF, "double", tmp_npf, "ny");
  // surface Jacobian / Jacobian (used when lifting the boundary fluxes)
  J          = op_decl_dat(cells, DG_NP, "double", tmp_np, "J");
  sJ         = op_decl_dat(cells, 3 * DG_NPF, "double", tmp_npf, "sJ");
  fscale     = op_decl_dat(cells, 3 * DG_NPF, "double", tmp_npf, "fscale");
  bool *bool_tmp_1_e = (bool *)calloc(edges->size, sizeof(bool));
  reverse = op_decl_dat(edges, 1, "bool", bool_tmp_1_e, "reverse");
  free(bool_tmp_1_e);
  int *int_tmp_1 = (int *)calloc(cells->size, sizeof(int));
  order = op_decl_dat(cells, 1, "int", int_tmp_1, "order");
  free(int_tmp_1);
  for(int i = 0; i < 4; i++) {
    string tmpname = "op_tmp" + to_string(i);
    op_tmp[i] = op_decl_dat(cells, DG_NP, "double", tmp_npf, tmpname.c_str());
  }
  free(tmp_npf);
  free(tmp_np);

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
