#include "dg_mesh/dg_mesh_3d.h"

#include "op_seq.h"

#include <iostream>

#include "dg_constants/dg_constants_3d.h"
#include "dg_compiler_defs.h"
#include "dg_op2_blas.h"
#include "dg_global_constants/dg_global_constants_3d.h"
#include "dg_dat_pool.h"
#include "dg_abort.h"

void init_op2_gemv();
void destroy_op2_gemv();

DGConstants *constants;
DGDatPool *dg_dat_pool;

DGMesh3D::DGMesh3D(std::string &meshFile) {
  init_op2_gemv();

  // Sets
  nodes  = op_decl_set_hdf5(meshFile.c_str(), "nodes");
  cells  = op_decl_set_hdf5(meshFile.c_str(), "cells");
  faces  = op_decl_set_hdf5(meshFile.c_str(), "faces");
  bfaces = op_decl_set_hdf5(meshFile.c_str(), "bfaces");

  // Maps
  cell2nodes  = op_decl_map_hdf5(cells, nodes, 4, meshFile.c_str(), "cell2nodes");
  face2nodes  = op_decl_map_hdf5(faces, nodes, 3, meshFile.c_str(), "face2nodes");
  face2cells  = op_decl_map_hdf5(faces, cells, 2, meshFile.c_str(), "face2cells");
  bface2nodes = op_decl_map_hdf5(bfaces, nodes, 3, meshFile.c_str(), "bface2nodes");
  bface2cells = op_decl_map_hdf5(bfaces, cells, 1, meshFile.c_str(), "bface2cells");

  // Dats
  node_coords  = op_decl_dat_hdf5(nodes, 3, DG_FP_STR, meshFile.c_str(), "node_coords");
  faceNum      = op_decl_dat_hdf5(faces, 2, "int", meshFile.c_str(), "faceNum");
  bfaceNum     = op_decl_dat_hdf5(bfaces, 1, "int", meshFile.c_str(), "bfaceNum");
  periodicFace = op_decl_dat_hdf5(faces, 1, "int", meshFile.c_str(), "periodicFace");

  nodeX = op_decl_dat(cells, 4, DG_FP_STR, (DG_FP *)NULL, "nodeX");
  nodeY = op_decl_dat(cells, 4, DG_FP_STR, (DG_FP *)NULL, "nodeY");
  nodeZ = op_decl_dat(cells, 4, DG_FP_STR, (DG_FP *)NULL, "nodeZ");

  x = op_decl_dat(cells, DG_NP, DG_FP_STR, (DG_FP *)NULL, "x");
  y = op_decl_dat(cells, DG_NP, DG_FP_STR, (DG_FP *)NULL, "y");
  z = op_decl_dat(cells, DG_NP, DG_FP_STR, (DG_FP *)NULL, "z");

  geof = op_decl_dat(cells, 10, DG_FP_STR, (DG_FP *)NULL, "geof");
  nx_c = op_decl_dat(cells, 4, DG_FP_STR, (DG_FP *)NULL, "nx_c");
  ny_c = op_decl_dat(cells, 4, DG_FP_STR, (DG_FP *)NULL, "ny_c");
  nz_c = op_decl_dat(cells, 4, DG_FP_STR, (DG_FP *)NULL, "nz_c");
  sJ_c = op_decl_dat(cells, 4, DG_FP_STR, (DG_FP *)NULL, "sJ_c");

  fmaskL = op_decl_dat(faces, DG_NPF, "int", (int *)NULL, "fmaskL");
  fmaskR = op_decl_dat(faces, DG_NPF, "int", (int *)NULL, "fmaskR");

  nx     = op_decl_dat(faces, 2, DG_FP_STR, (DG_FP *)NULL, "nx");
  ny     = op_decl_dat(faces, 2, DG_FP_STR, (DG_FP *)NULL, "ny");
  nz     = op_decl_dat(faces, 2, DG_FP_STR, (DG_FP *)NULL, "nz");
  sJ     = op_decl_dat(faces, 2, DG_FP_STR, (DG_FP *)NULL, "sJ");
  fscale = op_decl_dat(faces, 2, DG_FP_STR, (DG_FP *)NULL, "fscale");

  bnx     = op_decl_dat(bfaces, 1, DG_FP_STR, (DG_FP *)NULL, "bnx");
  bny     = op_decl_dat(bfaces, 1, DG_FP_STR, (DG_FP *)NULL, "bny");
  bnz     = op_decl_dat(bfaces, 1, DG_FP_STR, (DG_FP *)NULL, "bnz");
  bsJ     = op_decl_dat(bfaces, 1, DG_FP_STR, (DG_FP *)NULL, "bsJ");
  bfscale = op_decl_dat(bfaces, 1, DG_FP_STR, (DG_FP *)NULL, "bfscale");

  dg_dat_pool = new DGDatPool(this);

  constants = new DGConstants3D(DG_ORDER);
  constants->calc_interp_mats();

  op_decl_const(DG_ORDER * DG_NPF * 4, "int", FMASK_TK);
  op_decl_const(DG_ORDER * 2, "int", DG_CONSTANTS_TK);

  order_int = DG_ORDER;

  #ifdef USE_CUSTOM_MAPS
  for(int i = 0; i < DG_ORDER; i++) {
    custom_map_info tmp;
    node2node_custom_maps.push_back(tmp);
  }
  #endif
}


DGMesh3D::~DGMesh3D() {
  #ifdef USE_CUSTOM_MAPS
  for(int i = 0; i < DG_ORDER; i++) {
    free_custom_map(node2node_custom_maps[i]);
  }
  #endif
  delete dg_dat_pool;
  delete (DGConstants3D *)constants;
  destroy_op2_gemv();
}

void DGMesh3D::init() {
  constants->profile_blas(this);

  op_par_loop(init_nodes_3d, "init_nodes_3d", cells,
              op_arg_dat(node_coords, -4, cell2nodes, 3, DG_FP_STR, OP_READ),
              op_arg_dat(nodeX, -1, OP_ID, 4, DG_FP_STR, OP_WRITE),
              op_arg_dat(nodeY, -1, OP_ID, 4, DG_FP_STR, OP_WRITE),
              op_arg_dat(nodeZ, -1, OP_ID, 4, DG_FP_STR, OP_WRITE));

  op_par_loop(init_grid_3d, "init_grid_3d", cells,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(nodeX, -1, OP_ID, 4, DG_FP_STR, OP_READ),
              op_arg_dat(nodeY, -1, OP_ID, 4, DG_FP_STR, OP_READ),
              op_arg_dat(nodeZ, -1, OP_ID, 4, DG_FP_STR, OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(z, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  op_par_loop(init_geometric_factors_3d, "init_geometric_factors_3d", cells,
              op_arg_dat(x, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(z, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(geof, -1, OP_ID, 10, DG_FP_STR, OP_WRITE));

  op_par_loop(init_faces_3d, "init_faces_3d", faces,
              op_arg_dat(faceNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(geof, -2, face2cells, 10, DG_FP_STR, OP_READ),
              op_arg_dat(nx, -1, OP_ID, 2, DG_FP_STR, OP_WRITE),
              op_arg_dat(ny, -1, OP_ID, 2, DG_FP_STR, OP_WRITE),
              op_arg_dat(nz, -1, OP_ID, 2, DG_FP_STR, OP_WRITE),
              op_arg_dat(sJ, -1, OP_ID, 2, DG_FP_STR, OP_WRITE),
              op_arg_dat(fscale, -1, OP_ID, 2, DG_FP_STR, OP_WRITE));

  int num_norm = 0;
  op_par_loop(normals_check_3d, "normals_check_3d", faces,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(faceNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(nx, -1, OP_ID, 2, DG_FP_STR, OP_RW),
              op_arg_dat(ny, -1, OP_ID, 2, DG_FP_STR, OP_RW),
              op_arg_dat(nz, -1, OP_ID, 2, DG_FP_STR, OP_RW),
              op_arg_dat(x, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(y, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(z, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(nodeX, -2, face2cells, 4, DG_FP_STR, OP_READ),
              op_arg_dat(nodeY, -2, face2cells, 4, DG_FP_STR, OP_READ),
              op_arg_dat(nodeZ, -2, face2cells, 4, DG_FP_STR, OP_READ),
              op_arg_gbl(&num_norm, 1, "int", OP_INC));
  if(num_norm != 0) {
    dg_abort("Number of normal errors: " + std::to_string(num_norm));
  }

  op_par_loop(copy_normals_3d, "copy_normals_3d", faces,
              op_arg_dat(faceNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(nx, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(ny, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(nz, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(sJ, -1, OP_ID, 2, DG_FP_STR, OP_READ),
              op_arg_dat(nx_c, -2, face2cells, 4, DG_FP_STR, OP_WRITE),
              op_arg_dat(ny_c, -2, face2cells, 4, DG_FP_STR, OP_WRITE),
              op_arg_dat(nz_c, -2, face2cells, 4, DG_FP_STR, OP_WRITE),
              op_arg_dat(sJ_c, -2, face2cells, 4, DG_FP_STR, OP_WRITE));

  if(bface2cells) {
    op_par_loop(init_bfaces_3d, "init_bfaces_3d", bfaces,
                op_arg_dat(bfaceNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(geof, 0, bface2cells, 10, DG_FP_STR, OP_READ),
                op_arg_dat(bnx, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
                op_arg_dat(bny, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
                op_arg_dat(bnz, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
                op_arg_dat(bsJ, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
                op_arg_dat(bfscale, -1, OP_ID, 1, DG_FP_STR, OP_WRITE));
  }

  if(bface2cells) {
    op_par_loop(copy_normals_bfaces_3d, "copy_normals_bfaces_3d", bfaces,
                op_arg_dat(bfaceNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(bnx, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(bny, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(bnz, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(bsJ, -1, OP_ID, 1, DG_FP_STR, OP_READ),
                op_arg_dat(nx_c, 0, bface2cells, 4, DG_FP_STR, OP_WRITE),
                op_arg_dat(ny_c, 0, bface2cells, 4, DG_FP_STR, OP_WRITE),
                op_arg_dat(nz_c, 0, bface2cells, 4, DG_FP_STR, OP_WRITE),
                op_arg_dat(sJ_c, 0, bface2cells, 4, DG_FP_STR, OP_WRITE));
  }

  calc_mesh_constants();
}

void DGMesh3D::calc_mesh_constants() {
  op_par_loop(init_grid_3d, "init_grid_3d", cells,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(nodeX, -1, OP_ID, 4, DG_FP_STR, OP_READ),
              op_arg_dat(nodeY, -1, OP_ID, 4, DG_FP_STR, OP_READ),
              op_arg_dat(nodeZ, -1, OP_ID, 4, DG_FP_STR, OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(z, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  if(periodicFace) {
    int num = 0;
    op_par_loop(face_check_3d_periodic, "face_check_3d_periodic", faces,
                op_arg_gbl(&order_int, 1, "int", OP_READ),
                op_arg_dat(faceNum, -1, OP_ID, 2, "int", OP_READ),
                op_arg_dat(periodicFace, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(x, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(y, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(z, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(fmaskL, -1, OP_ID, DG_NPF, "int", OP_WRITE),
                op_arg_dat(fmaskR, -1, OP_ID, DG_NPF, "int", OP_WRITE),
                op_arg_gbl(&num, 1, "int", OP_INC));
    if(num > 0) {
      dg_abort("Number of non matching points on faces: " + std::to_string(num));
    }
  } else {
    int num = 0;
    op_par_loop(face_check_3d, "face_check_3d", faces,
                op_arg_gbl(&order_int, 1, "int", OP_READ),
                op_arg_dat(faceNum, -1, OP_ID, 2, "int", OP_READ),
                op_arg_dat(x, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(y, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(z, -2, face2cells, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(fmaskL, -1, OP_ID, DG_NPF, "int", OP_WRITE),
                op_arg_dat(fmaskR, -1, OP_ID, DG_NPF, "int", OP_WRITE),
                op_arg_gbl(&num, 1, "int", OP_INC));
    if(num > 0) {
      dg_abort("Number of non matching points on faces: " + std::to_string(num));
    }
  }

  #ifdef USE_CUSTOM_MAPS
  update_custom_map();
  #endif
}

void DGMesh3D::update_order(int new_order, std::vector<op_dat> &dats_to_interp) {
  for(int i = 0; i < dats_to_interp.size(); i++) {
    if(dats_to_interp[i]->dim != DG_NP) {
      dg_abort("Interpolating between orders for non DG_NP dim dats is not implemented ...  exiting");
    }

    interp_dat_between_orders(order_int, new_order, dats_to_interp[i]);
  }

  order_int = new_order;

  calc_mesh_constants();
}

void DGMesh3D::update_order_sp(int new_order, std::vector<op_dat> &dats_to_interp) {
  for(int i = 0; i < dats_to_interp.size(); i++) {
    if(dats_to_interp[i]->dim != DG_NP) {
      dg_abort("Interpolating between orders for non DG_NP dim dats is not implemented ...  exiting");
    }

    interp_dat_between_orders_sp(order_int, new_order, dats_to_interp[i]);
  }

  order_int = new_order;

  calc_mesh_constants();
}
