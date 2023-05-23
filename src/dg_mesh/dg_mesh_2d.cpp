#include "dg_mesh/dg_mesh_2d.h"

#include "op_seq.h"

#include <string>
#include <memory>
#include <iostream>

#include "dg_compiler_defs.h"
#include "dg_op2_blas.h"
#include "dg_dat_pool.h"

#include "dg_constants/dg_constants_2d.h"
#include "dg_global_constants/dg_global_constants_2d.h"

DGConstants *constants;
DGDatPool *dg_dat_pool;

using namespace std;

DGCubatureData::DGCubatureData(DGMesh2D *m) {
  mesh = m;

  DG_FP *tmp_cub_np = (DG_FP *)calloc(DG_CUB_NP * mesh->cells->size, sizeof(DG_FP));
  DG_FP *tmp_np_np = (DG_FP *)calloc(DG_NP * DG_NP * mesh->cells->size, sizeof(DG_FP));
  DG_FP *tmp_cub_np_np = (DG_FP *)calloc(DG_CUB_NP * DG_NP * mesh->cells->size, sizeof(DG_FP));
  rx    = op_decl_dat(mesh->cells, DG_CUB_NP, DG_FP_STR, tmp_cub_np, "cub-rx");
  sx    = op_decl_dat(mesh->cells, DG_CUB_NP, DG_FP_STR, tmp_cub_np, "cub-sx");
  ry    = op_decl_dat(mesh->cells, DG_CUB_NP, DG_FP_STR, tmp_cub_np, "cub-ry");
  sy    = op_decl_dat(mesh->cells, DG_CUB_NP, DG_FP_STR, tmp_cub_np, "cub-sy");
  J     = op_decl_dat(mesh->cells, DG_CUB_NP, DG_FP_STR, tmp_cub_np, "cub-J");
  mm    = op_decl_dat(mesh->cells, DG_NP * DG_NP, DG_FP_STR, tmp_np_np, "cub-mm");
  tmp   = op_decl_dat(mesh->cells, DG_CUB_NP * DG_NP, DG_FP_STR, tmp_cub_np_np, "cub-tmp");

  for(int i = 0; i < 4; i++) {
    string tmpname = "cub-op_tmp" + to_string(i);
    op_tmp[i] = op_decl_dat(mesh->cells, DG_CUB_NP, DG_FP_STR, tmp_cub_np, tmpname.c_str());
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
              op_arg_gbl(constants->get_mat_ptr(DGConstants::CUB_V), DG_ORDER * DG_CUB_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rx,    -1, OP_ID, DG_CUB_NP, DG_FP_STR, OP_RW),
              op_arg_dat(sx,    -1, OP_ID, DG_CUB_NP, DG_FP_STR, OP_RW),
              op_arg_dat(ry,    -1, OP_ID, DG_CUB_NP, DG_FP_STR, OP_RW),
              op_arg_dat(sy,    -1, OP_ID, DG_CUB_NP, DG_FP_STR, OP_RW),
              op_arg_dat(J,     -1, OP_ID, DG_CUB_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(tmp,   -1, OP_ID, DG_CUB_NP * DG_NP, DG_FP_STR, OP_WRITE));

  op_par_loop(cub_mm_init, "cub_mm_init", mesh->cells,
              op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::CUB_V), DG_ORDER * DG_CUB_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp,   -1, OP_ID, DG_CUB_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mm,    -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_WRITE));
}

DGGaussData::DGGaussData(DGMesh2D *m) {
  mesh = m;

  DG_FP *tmp_g_np = (DG_FP *)calloc(DG_G_NP * mesh->cells->size, sizeof(DG_FP));
  x  = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, tmp_g_np, "gauss-x");
  y  = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, tmp_g_np, "gauss-y");
  rx = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, tmp_g_np, "gauss-rx");
  sx = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, tmp_g_np, "gauss-sx");
  ry = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, tmp_g_np, "gauss-ry");
  sy = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, tmp_g_np, "gauss-sy");
  sJ = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, tmp_g_np, "gauss-sJ");
  nx = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, tmp_g_np, "gauss-nx");
  ny = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, tmp_g_np, "gauss-ny");

  for(int i = 0; i < 3; i++) {
    string tmpname = "gauss-op_tmp" + to_string(i);
    op_tmp[i] = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, tmp_g_np, tmpname.c_str());
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
              op_arg_dat(mesh->x, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->y, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F0DR), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F0DS), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F1DR), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F1DS), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F2DR), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_F2DS), DG_ORDER * DG_GF_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rx, -1, OP_ID, DG_G_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(sx, -1, OP_ID, DG_G_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(ry, -1, OP_ID, DG_G_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(sy, -1, OP_ID, DG_G_NP, DG_FP_STR, OP_WRITE));

  op_par_loop(init_gauss, "init_gauss", mesh->cells,
              op_arg_dat(mesh->order,  -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(rx,     -1, OP_ID, DG_G_NP, DG_FP_STR, OP_RW),
              op_arg_dat(sx,     -1, OP_ID, DG_G_NP, DG_FP_STR, OP_RW),
              op_arg_dat(ry,     -1, OP_ID, DG_G_NP, DG_FP_STR, OP_RW),
              op_arg_dat(sy,     -1, OP_ID, DG_G_NP, DG_FP_STR, OP_RW),
              op_arg_dat(nx,     -1, OP_ID, DG_G_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(ny,     -1, OP_ID, DG_G_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(sJ,     -1, OP_ID, DG_G_NP, DG_FP_STR, OP_WRITE));

  int num_norm = 0;
  op_par_loop(normals_check_2d, "normals_check_2d", mesh->faces,
              op_arg_dat(mesh->edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(mesh->reverse, -1, OP_ID, 1, "bool", OP_READ),
              op_arg_dat(nx, -2, mesh->face2cells, DG_G_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ny, -2, mesh->face2cells, DG_G_NP, DG_FP_STR, OP_READ),
              op_arg_dat(x,  -2, mesh->face2cells, DG_G_NP, DG_FP_STR, OP_READ),
              op_arg_dat(y,  -2, mesh->face2cells, DG_G_NP, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->nodeX, -2, mesh->face2cells, 3, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->nodeY, -2, mesh->face2cells, 3, DG_FP_STR, OP_READ),
              op_arg_gbl(&num_norm, 1, "int", OP_INC));
  if(num_norm != 0) {
    std::cout << "Number of normal errors: " << num_norm << std::endl;
    exit(-1);
  }
}

DGMesh2D::DGMesh2D(std::string &meshFile) {
  // Calculate DG constants
  constants = new DGConstants2D(DG_ORDER);

  constants->calc_interp_mats();

  // Initialise OP2
  // Declare OP2 sets
  nodes  = op_decl_set_hdf5(meshFile.c_str(), "nodes");
  cells  = op_decl_set_hdf5(meshFile.c_str(), "cells");
  faces  = op_decl_set_hdf5(meshFile.c_str(), "faces");
  bfaces = op_decl_set_hdf5(meshFile.c_str(), "bfaces");

  // Declare OP2 maps
  cell2nodes  = op_decl_map_hdf5(cells, nodes, 3, meshFile.c_str(), "cell2nodes");
  face2nodes  = op_decl_map_hdf5(faces, nodes, 2, meshFile.c_str(), "face2nodes");
  face2cells  = op_decl_map_hdf5(faces, cells, 2, meshFile.c_str(), "face2cells");
  bface2nodes = op_decl_map_hdf5(bfaces, nodes, 2, meshFile.c_str(), "bface2nodes");
  bface2cells = op_decl_map_hdf5(bfaces, cells, 1, meshFile.c_str(), "bface2cells");

  // Declare OP2 datasets from HDF5
  // Structure: {x, y}
  node_coords = op_decl_dat_hdf5(nodes, 2, DG_FP_STR, meshFile.c_str(), "node_coords");
  bedge_type  = op_decl_dat_hdf5(bfaces, 1, "int", meshFile.c_str(), "bedge_type");
  edgeNum     = op_decl_dat_hdf5(faces, 2, "int", meshFile.c_str(), "edgeNum");
  bedgeNum    = op_decl_dat_hdf5(bfaces, 1, "int", meshFile.c_str(), "bedgeNum");

  // Declare regular OP2 datasets
  // Coords of nodes per cell
  DG_FP *tmp_3_data = (DG_FP*)calloc(3 * cells->size, sizeof(DG_FP));
  nodeX = op_decl_dat(cells, 3, DG_FP_STR, tmp_3_data, "nodeX");
  nodeY = op_decl_dat(cells, 3, DG_FP_STR, tmp_3_data, "nodeY");
  free(tmp_3_data);

  DG_FP *tmp_np  = (DG_FP *)calloc(DG_NP * cells->size, sizeof(DG_FP));
  DG_FP *tmp_npf = (DG_FP *)calloc(DG_NUM_FACES * DG_NPF * cells->size, sizeof(DG_FP));
  // The x and y coordinates of all the solution points in a cell
  x = op_decl_dat(cells, DG_NP, DG_FP_STR, tmp_np, "x");
  y = op_decl_dat(cells, DG_NP, DG_FP_STR, tmp_np, "y");
  // Geometric factors that relate to mapping between global and local (cell) coordinates
  rx = op_decl_dat(cells, DG_NP, DG_FP_STR, tmp_np, "rx");
  ry = op_decl_dat(cells, DG_NP, DG_FP_STR, tmp_np, "ry");
  sx = op_decl_dat(cells, DG_NP, DG_FP_STR, tmp_np, "sx");
  sy = op_decl_dat(cells, DG_NP, DG_FP_STR, tmp_np, "sy");
  // Normals for each cell (calculated for each node on each edge, nodes can appear on multiple edges)
  nx = op_decl_dat(cells, 3 * DG_NPF, DG_FP_STR, tmp_npf, "nx");
  ny = op_decl_dat(cells, 3 * DG_NPF, DG_FP_STR, tmp_npf, "ny");
  // surface Jacobian / Jacobian (used when lifting the boundary fluxes)
  J          = op_decl_dat(cells, DG_NP, DG_FP_STR, tmp_np, "J");
  sJ         = op_decl_dat(cells, 3 * DG_NPF, DG_FP_STR, tmp_npf, "sJ");
  fscale     = op_decl_dat(cells, 3 * DG_NPF, DG_FP_STR, tmp_npf, "fscale");
  bool *bool_tmp_1_e = (bool *)calloc(faces->size, sizeof(bool));
  reverse = op_decl_dat(faces, 1, "bool", bool_tmp_1_e, "reverse");
  free(bool_tmp_1_e);
  int *int_tmp_1 = (int *)calloc(cells->size, sizeof(int));
  order = op_decl_dat(cells, 1, "int", int_tmp_1, "order");
  free(int_tmp_1);
  for(int i = 0; i < 4; i++) {
    string tmpname = "op_tmp" + to_string(i);
    op_tmp[i] = op_decl_dat(cells, DG_NP, DG_FP_STR, tmp_npf, tmpname.c_str());
  }
  free(tmp_npf);
  free(tmp_np);

  dg_dat_pool = new DGDatPool(this);

  op_decl_const(DG_ORDER * 5, "int", DG_CONSTANTS_TK);
  op_decl_const(DG_ORDER * DG_NPF * 3, "int", FMASK_TK);
  op_decl_const(DG_ORDER * DG_CUB_NP, DG_FP_STR, cubW_g_TK);
  op_decl_const(DG_ORDER * DG_GF_NP, DG_FP_STR, gaussW_g_TK);

  cubature = new DGCubatureData(this);
  gauss = new DGGaussData(this);
}

DGMesh2D::~DGMesh2D() {
  delete constants;
  delete cubature;
  delete gauss;
}

void DGMesh2D::init() {
  // Initialise the order to the max order to start with
  op_par_loop(init_order, "init_order", cells,
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_WRITE));

  op_par_loop(init_nodes, "init_nodes", cells,
              op_arg_dat(node_coords, -3, cell2nodes, 2, DG_FP_STR, OP_READ),
              op_arg_dat(nodeX, -1, OP_ID, 3, DG_FP_STR, OP_WRITE),
              op_arg_dat(nodeY, -1, OP_ID, 3, DG_FP_STR, OP_WRITE));

  // Calculate geometric factors
  op_par_loop(calc_geom, "calc_geom", cells,
              op_arg_dat(order,  -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::R), DG_ORDER * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::S), DG_ORDER * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(nodeX, -1, OP_ID, 3, DG_FP_STR, OP_READ),
              op_arg_dat(nodeY, -1, OP_ID, 3, DG_FP_STR, OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::DR, x, 0.0, rx);
  op2_gemv(this, false, 1.0, DGConstants::DS, x, 0.0, sx);
  op2_gemv(this, false, 1.0, DGConstants::DR, y, 0.0, ry);
  op2_gemv(this, false, 1.0, DGConstants::DS, y, 0.0, sy);

  op_par_loop(init_grid, "init_grid", cells,
              op_arg_dat(order,  -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(rx,     -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(ry,     -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(sx,     -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(sy,     -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(nx,     -1, OP_ID, 3 * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(ny,     -1, OP_ID, 3 * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(J,      -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(sJ,     -1, OP_ID, 3 * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(fscale, -1, OP_ID, 3 * DG_NPF, DG_FP_STR, OP_WRITE));

  op_par_loop(init_edges, "init_edges", faces,
              op_arg_dat(edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(nodeX, -2, face2cells, 3, DG_FP_STR, OP_READ),
              op_arg_dat(nodeY, -2, face2cells, 3, DG_FP_STR, OP_READ),
              op_arg_dat(reverse, -1, OP_ID, 1, "bool", OP_WRITE));

  cubature->init();
  gauss->init();
}

void DGMesh2D::update_mesh_constants() {
  op_par_loop(calc_geom, "calc_geom", cells,
              op_arg_dat(order,  -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::R), DG_ORDER * DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(constants->get_mat_ptr(DGConstants::S), DG_ORDER * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(nodeX, -1, OP_ID, 3, DG_FP_STR, OP_READ),
              op_arg_dat(nodeY, -1, OP_ID, 3, DG_FP_STR, OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::DR, x, 0.0, rx);
  op2_gemv(this, false, 1.0, DGConstants::DS, x, 0.0, sx);
  op2_gemv(this, false, 1.0, DGConstants::DR, y, 0.0, ry);
  op2_gemv(this, false, 1.0, DGConstants::DS, y, 0.0, sy);

  op_par_loop(init_grid, "init_grid", cells,
              op_arg_dat(order,  -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(rx,     -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(ry,     -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(sx,     -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(sy,     -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(nx,     -1, OP_ID, 3 * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(ny,     -1, OP_ID, 3 * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(J,      -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(sJ,     -1, OP_ID, 3 * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(fscale, -1, OP_ID, 3 * DG_NPF, DG_FP_STR, OP_WRITE));

  cubature->update_mesh_constants();
  gauss->update_mesh_constants();
}

void DGMesh2D::update_order(op_dat new_orders, std::vector<op_dat> &dats_to_interp) {
  // Interpolate dats first (assumes all these dats are of size DG_NP)
  for(int i = 0; i < dats_to_interp.size(); i++) {
    if(dats_to_interp[i]->dim != DG_NP) {
      std::cerr << "Interpolating between orders for non DG_NP dim dats is not implemented ...  exiting" << std::endl;
      exit(-1);
    }
    op_par_loop(interp_dat_to_new_order, "interp_dat_to_new_order", cells,
                op_arg_gbl(constants->get_mat_ptr(DGConstants::INTERP_MATRIX_ARRAY), DG_ORDER * DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(order,      -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(new_orders, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(dats_to_interp[i], -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  }

  // Copy across new orders
  op_par_loop(copy_new_orders, "copy_new_orders", cells,
              op_arg_dat(new_orders,  -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(order,  -1, OP_ID, 1, "int", OP_WRITE));

  // Update mesh constants for new orders
  update_mesh_constants();
}

void DGMesh2D::update_order(int new_order, std::vector<op_dat> &dats_to_interp) {
  // Interpolate dats first (assumes all these dats are of size DG_NP)
  for(int i = 0; i < dats_to_interp.size(); i++) {
    if(dats_to_interp[i]->dim != DG_NP) {
      std::cerr << "Interpolating between orders for non DG_NP dim dats is not implemented ...  exiting" << std::endl;
      exit(-1);
    }
    op_par_loop(interp_dat_to_new_order_int, "interp_dat_to_new_order_int", cells,
                op_arg_gbl(constants->get_mat_ptr(DGConstants::INTERP_MATRIX_ARRAY), DG_ORDER * DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(order,     -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&new_order, 1, "int", OP_READ),
                op_arg_dat(dats_to_interp[i], -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  }

  // Copy across new orders
  op_par_loop(copy_new_orders_int, "copy_new_orders_int", cells,
              op_arg_gbl(&new_order, 1, "int", OP_READ),
              op_arg_dat(order,  -1, OP_ID, 1, "int", OP_WRITE));

  // Update mesh constants for new orders
  update_mesh_constants();
}

void DGMesh2D::interp_to_max_order(std::vector<op_dat> &dats_in,
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
                op_arg_gbl(constants->get_mat_ptr(DGConstants::INTERP_MATRIX_ARRAY), DG_ORDER * DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(order,       -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(dats_in[i],  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(dats_out[i], -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));
  }
}
