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

  rx  = op_decl_dat(mesh->cells, DG_CUB_NP, DG_FP_STR, (DG_FP *)NULL, "cub-rx");
  sx  = op_decl_dat(mesh->cells, DG_CUB_NP, DG_FP_STR, (DG_FP *)NULL, "cub-sx");
  ry  = op_decl_dat(mesh->cells, DG_CUB_NP, DG_FP_STR, (DG_FP *)NULL, "cub-ry");
  sy  = op_decl_dat(mesh->cells, DG_CUB_NP, DG_FP_STR, (DG_FP *)NULL, "cub-sy");
  J   = op_decl_dat(mesh->cells, DG_CUB_NP, DG_FP_STR, (DG_FP *)NULL, "cub-J");
  mm  = op_decl_dat(mesh->cells, DG_NP * DG_NP, DG_FP_STR, (DG_FP *)NULL, "cub-mm");
  tmp = op_decl_dat(mesh->cells, DG_CUB_NP * DG_NP, DG_FP_STR, (DG_FP *)NULL, "cub-tmp");

  for(int i = 0; i < 4; i++) {
    string tmpname = "cub-op_tmp" + to_string(i);
    op_tmp[i] = op_decl_dat(mesh->cells, DG_CUB_NP, DG_FP_STR, (DG_FP *)NULL, tmpname.c_str());
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

  x  = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, (DG_FP *)NULL, "gauss-x");
  y  = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, (DG_FP *)NULL, "gauss-y");
  rx = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, (DG_FP *)NULL, "gauss-rx");
  sx = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, (DG_FP *)NULL, "gauss-sx");
  ry = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, (DG_FP *)NULL, "gauss-ry");
  sy = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, (DG_FP *)NULL, "gauss-sy");
  sJ = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, (DG_FP *)NULL, "gauss-sJ");
  nx = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, (DG_FP *)NULL, "gauss-nx");
  ny = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, (DG_FP *)NULL, "gauss-ny");

  for(int i = 0; i < 3; i++) {
    string tmpname = "gauss-op_tmp" + to_string(i);
    op_tmp[i] = op_decl_dat(mesh->cells, DG_G_NP, DG_FP_STR, (DG_FP *)NULL, tmpname.c_str());
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

DGMesh2D::DGMesh2D(std::string &meshFile, bool overInt) {
  over_integrate = overInt;
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
  nodeX = op_decl_dat(cells, 3, DG_FP_STR, (DG_FP *)NULL, "nodeX");
  nodeY = op_decl_dat(cells, 3, DG_FP_STR, (DG_FP *)NULL, "nodeY");

  // The x and y coordinates of all the solution points in a cell
  x = op_decl_dat(cells, DG_NP, DG_FP_STR, (DG_FP *)NULL, "x");
  y = op_decl_dat(cells, DG_NP, DG_FP_STR, (DG_FP *)NULL, "y");
  // Geometric factors that relate to mapping between global and local (cell) coordinates
  rx = op_decl_dat(cells, DG_NP, DG_FP_STR, (DG_FP *)NULL, "rx");
  ry = op_decl_dat(cells, DG_NP, DG_FP_STR, (DG_FP *)NULL, "ry");
  sx = op_decl_dat(cells, DG_NP, DG_FP_STR, (DG_FP *)NULL, "sx");
  sy = op_decl_dat(cells, DG_NP, DG_FP_STR, (DG_FP *)NULL, "sy");
  geof = op_decl_dat(cells, 5, DG_FP_STR, (DG_FP *)NULL, "geof");
  // Normals for each cell (calculated for each node on each edge, nodes can appear on multiple edges)
  nx_c = op_decl_dat(cells, 3 * DG_NPF, DG_FP_STR, (DG_FP *)NULL, "nx_c");
  ny_c = op_decl_dat(cells, 3 * DG_NPF, DG_FP_STR, (DG_FP *)NULL, "ny_c");
  // surface Jacobian / Jacobian (used when lifting the boundary fluxes)
  J        = op_decl_dat(cells, DG_NP, DG_FP_STR, (DG_FP *)NULL, "J");
  sJ_c     = op_decl_dat(cells, 3 * DG_NPF, DG_FP_STR, (DG_FP *)NULL, "sJ_c");
  fscale_c = op_decl_dat(cells, 3 * DG_NPF, DG_FP_STR, (DG_FP *)NULL, "fscale_c");
  reverse = op_decl_dat(faces, 1, "bool", (bool *)NULL, "reverse");
  nx = op_decl_dat(faces, 2, DG_FP_STR, (DG_FP *)NULL, "nx");
  ny = op_decl_dat(faces, 2, DG_FP_STR, (DG_FP *)NULL, "ny");
  sJ = op_decl_dat(faces, 2, DG_FP_STR, (DG_FP *)NULL, "sJ");
  fscale = op_decl_dat(faces, 2, DG_FP_STR, (DG_FP *)NULL, "fscale");
  bnx = op_decl_dat(bfaces, 1, DG_FP_STR, (DG_FP *)NULL, "bnx");
  bny = op_decl_dat(bfaces, 1, DG_FP_STR, (DG_FP *)NULL, "bny");
  bsJ = op_decl_dat(bfaces, 1, DG_FP_STR, (DG_FP *)NULL, "bsJ");
  bfscale = op_decl_dat(bfaces, 1, DG_FP_STR, (DG_FP *)NULL, "bfscale");
  order = op_decl_dat(cells, 1, "int", (int *)NULL, "order");
  for(int i = 0; i < 4; i++) {
    string tmpname = "op_tmp" + to_string(i);
    op_tmp[i] = op_decl_dat(cells, DG_NP, DG_FP_STR, (DG_FP *)NULL, tmpname.c_str());
  }

  dg_dat_pool = new DGDatPool(this);

  op_decl_const(DG_ORDER * 5, "int", DG_CONSTANTS_TK);
  op_decl_const(DG_ORDER * DG_NPF * 3, "int", FMASK_TK);
  op_decl_const(DG_ORDER * DG_CUB_NP, DG_FP_STR, cubW_g_TK);
  op_decl_const(DG_ORDER * DG_GF_NP, DG_FP_STR, gaussW_g_TK);

  if(over_integrate) {
    cubature = new DGCubatureData(this);
    gauss = new DGGaussData(this);
  }

  order_int = DG_ORDER;
}

DGMesh2D::~DGMesh2D() {
  delete constants;
  if(over_integrate) {
    delete cubature;
    delete gauss;
  }
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
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(rx,    -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(ry,    -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(sx,    -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(sy,    -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(nx_c,  -1, OP_ID, 3 * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(ny_c,  -1, OP_ID, 3 * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(J,     -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(sJ_c,  -1, OP_ID, 3 * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(fscale_c, -1, OP_ID, 3 * DG_NPF, DG_FP_STR, OP_WRITE));

  op_par_loop(init_edges, "init_edges", faces,
              op_arg_dat(edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(nodeX, -2, face2cells, 3, DG_FP_STR, OP_READ),
              op_arg_dat(nodeY, -2, face2cells, 3, DG_FP_STR, OP_READ),
              op_arg_dat(reverse, -1, OP_ID, 1, "bool", OP_WRITE));

  op_par_loop(copy_normals_2d, "copy_normals_2d", faces,
              op_arg_dat(order, -2, face2cells, 1, "int", OP_READ),
              op_arg_dat(edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(nx_c, -2, face2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
              op_arg_dat(ny_c, -2, face2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
              op_arg_dat(sJ_c, -2, face2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
              op_arg_dat(fscale_c, -2, face2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
              op_arg_dat(nx, -1, OP_ID, 2, DG_FP_STR, OP_WRITE),
              op_arg_dat(ny, -1, OP_ID, 2, DG_FP_STR, OP_WRITE),
              op_arg_dat(sJ, -1, OP_ID, 2, DG_FP_STR, OP_WRITE),
              op_arg_dat(fscale, -1, OP_ID, 2, DG_FP_STR, OP_WRITE));

  if(bface2cells) {
    op_par_loop(copy_normals_bface_2d, "copy_normals_bface_2d", bfaces,
                op_arg_dat(order, 0, bface2cells, 1, "int", OP_READ),
                op_arg_dat(bedgeNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(nx_c, 0, bface2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
                op_arg_dat(ny_c, 0, bface2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
                op_arg_dat(sJ_c, 0, bface2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
                op_arg_dat(fscale_c, 0, bface2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
                op_arg_dat(bnx, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
                op_arg_dat(bny, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
                op_arg_dat(bsJ, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
                op_arg_dat(bfscale, -1, OP_ID, 1, DG_FP_STR, OP_WRITE));
  }

  op_par_loop(copy_geof_2d, "copy_geof_2d", cells,
              op_arg_dat(rx,    -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ry,    -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(sx,    -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(sy,    -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(J,     -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(geof,  -1, OP_ID, 5, DG_FP_STR, OP_WRITE));

  if(over_integrate) {
    cubature->init();
    gauss->init();
  }
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
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(rx,    -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(ry,    -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(sx,    -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(sy,    -1, OP_ID, DG_NP, DG_FP_STR, OP_RW),
              op_arg_dat(nx_c,  -1, OP_ID, 3 * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(ny_c,  -1, OP_ID, 3 * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(J,     -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(sJ_c,  -1, OP_ID, 3 * DG_NPF, DG_FP_STR, OP_WRITE),
              op_arg_dat(fscale_c, -1, OP_ID, 3 * DG_NPF, DG_FP_STR, OP_WRITE));

  op_par_loop(copy_normals_2d, "copy_normals_2d", faces,
              op_arg_dat(order, -2, face2cells, 1, "int", OP_READ),
              op_arg_dat(edgeNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(nx_c, -2, face2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
              op_arg_dat(ny_c, -2, face2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
              op_arg_dat(sJ_c, -2, face2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
              op_arg_dat(fscale_c, -2, face2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
              op_arg_dat(nx, -1, OP_ID, 2, DG_FP_STR, OP_WRITE),
              op_arg_dat(ny, -1, OP_ID, 2, DG_FP_STR, OP_WRITE),
              op_arg_dat(sJ, -1, OP_ID, 2, DG_FP_STR, OP_WRITE),
              op_arg_dat(fscale, -1, OP_ID, 2, DG_FP_STR, OP_WRITE));

  if(bface2cells) {
    op_par_loop(copy_normals_bface_2d, "copy_normals_bface_2d", bfaces,
                op_arg_dat(order, 0, bface2cells, 1, "int", OP_READ),
                op_arg_dat(bedgeNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(nx_c, 0, bface2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
                op_arg_dat(ny_c, 0, bface2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
                op_arg_dat(sJ_c, 0, bface2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
                op_arg_dat(fscale_c, 0, bface2cells, 3 * DG_NPF, DG_FP_STR, OP_READ),
                op_arg_dat(bnx, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
                op_arg_dat(bny, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
                op_arg_dat(bsJ, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
                op_arg_dat(bfscale, -1, OP_ID, 1, DG_FP_STR, OP_WRITE));
  }

  op_par_loop(copy_geof_2d, "copy_geof_2d", cells,
              op_arg_dat(rx,    -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ry,    -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(sx,    -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(sy,    -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(J,     -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(geof,  -1, OP_ID, 5, DG_FP_STR, OP_WRITE));

  if(over_integrate) {
    cubature->update_mesh_constants();
    gauss->update_mesh_constants();
  }
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

  order_int = new_order;

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
