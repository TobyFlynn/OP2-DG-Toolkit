#include "dg_mesh/dg_mesh_3d.h"

#include "op_seq.h"

#include <iostream>

#include "dg_constants/dg_constants_3d.h"
#include "dg_compiler_defs.h"
#include "dg_op2_blas.h"
#include "dg_global_constants/dg_global_constants_3d.h"

DGConstants *constants;

DGMesh3D::DGMesh3D(std::string &meshFile) {
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

  DG_FP *tmp_4 = (DG_FP *)calloc(4 * cells->size, sizeof(DG_FP));
  nodeX = op_decl_dat(cells, 4, DG_FP_STR, tmp_4, "nodeX");
  nodeY = op_decl_dat(cells, 4, DG_FP_STR, tmp_4, "nodeY");
  nodeZ = op_decl_dat(cells, 4, DG_FP_STR, tmp_4, "nodeZ");
  free(tmp_4);

  DG_FP *tmp_dg_np = (DG_FP *)calloc(DG_NP * cells->size, sizeof(DG_FP));
  x = op_decl_dat(cells, DG_NP, DG_FP_STR, tmp_dg_np, "x");
  y = op_decl_dat(cells, DG_NP, DG_FP_STR, tmp_dg_np, "y");
  z = op_decl_dat(cells, DG_NP, DG_FP_STR, tmp_dg_np, "z");
  for(int i = 0; i < 3; i++) {
    std::string tmpname = "op_tmp" + std::to_string(i);
    op_tmp[i] = op_decl_dat(cells, DG_NP, DG_FP_STR, tmp_dg_np, tmpname.c_str());
  }
  free(tmp_dg_np);

  DG_FP *tmp_1_cells_DG_FP = (DG_FP *)calloc(cells->size, sizeof(DG_FP));
  rx = op_decl_dat(cells, 1, DG_FP_STR, tmp_1_cells_DG_FP, "rx");
  ry = op_decl_dat(cells, 1, DG_FP_STR, tmp_1_cells_DG_FP, "ry");
  rz = op_decl_dat(cells, 1, DG_FP_STR, tmp_1_cells_DG_FP, "rz");
  sx = op_decl_dat(cells, 1, DG_FP_STR, tmp_1_cells_DG_FP, "sx");
  sy = op_decl_dat(cells, 1, DG_FP_STR, tmp_1_cells_DG_FP, "sy");
  sz = op_decl_dat(cells, 1, DG_FP_STR, tmp_1_cells_DG_FP, "sz");
  tx = op_decl_dat(cells, 1, DG_FP_STR, tmp_1_cells_DG_FP, "tx");
  ty = op_decl_dat(cells, 1, DG_FP_STR, tmp_1_cells_DG_FP, "ty");
  tz = op_decl_dat(cells, 1, DG_FP_STR, tmp_1_cells_DG_FP, "tz");
  J  = op_decl_dat(cells, 1, DG_FP_STR, tmp_1_cells_DG_FP, "J");
  free(tmp_1_cells_DG_FP);

  int *tmp_dg_npf_int = (int *)calloc(DG_NPF * faces->size, sizeof(int));
  fmaskL = op_decl_dat(faces, DG_NPF, "int", tmp_dg_npf_int, "fmaskL");
  fmaskR = op_decl_dat(faces, DG_NPF, "int", tmp_dg_npf_int, "fmaskR");
  free(tmp_dg_npf_int);

  DG_FP *tmp_1_faces_DG_FP = (DG_FP *)calloc(2 * faces->size, sizeof(DG_FP));
  nx     = op_decl_dat(faces, 2, DG_FP_STR, tmp_1_faces_DG_FP, "nx");
  ny     = op_decl_dat(faces, 2, DG_FP_STR, tmp_1_faces_DG_FP, "ny");
  nz     = op_decl_dat(faces, 2, DG_FP_STR, tmp_1_faces_DG_FP, "nz");
  sJ     = op_decl_dat(faces, 2, DG_FP_STR, tmp_1_faces_DG_FP, "sJ");
  fscale = op_decl_dat(faces, 2, DG_FP_STR, tmp_1_faces_DG_FP, "fscale");
  free(tmp_1_faces_DG_FP);
  DG_FP *tmp_1_bfaces_DG_FP = (DG_FP *)calloc(bfaces->size, sizeof(DG_FP));
  bnx     = op_decl_dat(bfaces, 1, DG_FP_STR, tmp_1_bfaces_DG_FP, "bnx");
  bny     = op_decl_dat(bfaces, 1, DG_FP_STR, tmp_1_bfaces_DG_FP, "bny");
  bnz     = op_decl_dat(bfaces, 1, DG_FP_STR, tmp_1_bfaces_DG_FP, "bnz");
  bsJ     = op_decl_dat(bfaces, 1, DG_FP_STR, tmp_1_bfaces_DG_FP, "bsJ");
  bfscale = op_decl_dat(bfaces, 1, DG_FP_STR, tmp_1_bfaces_DG_FP, "bfscale");
  free(tmp_1_bfaces_DG_FP);
  DG_FP *tmp_dg_npf_DG_FP = (DG_FP *)calloc(4 * DG_NPF * cells->size, sizeof(DG_FP));
  for(int i = 0; i < 3; i++) {
    std::string tmpname = "op_tmp_npf" + std::to_string(i);
    op_tmp_npf[i] = op_decl_dat(cells, 4 * DG_NPF, DG_FP_STR, tmp_dg_npf_DG_FP, tmpname.c_str());
  }
  free(tmp_dg_npf_DG_FP);
  int *int_tmp_1 = (int *)calloc(cells->size, sizeof(int));
  order = op_decl_dat(cells, 1, "int", int_tmp_1, "order");
  free(int_tmp_1);

  constants = new DGConstants3D(DG_ORDER);
  constants->calc_interp_mats();

  op_decl_const(DG_ORDER * DG_NPF * 4, "int", FMASK_TK);
  op_decl_const(DG_ORDER * 2, "int", DG_CONSTANTS_TK);

  order_int = DG_ORDER;
}


DGMesh3D::~DGMesh3D() {
  delete constants;
}

void DGMesh3D::init() {
  // Initialise the order to the max order to start with
  op_par_loop(init_order, "init_order", cells,
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_WRITE));

  op_par_loop(init_nodes_3d, "init_nodes_3d", cells,
              op_arg_dat(node_coords, -4, cell2nodes, 3, DG_FP_STR, OP_READ),
              op_arg_dat(nodeX, -1, OP_ID, 4, DG_FP_STR, OP_WRITE),
              op_arg_dat(nodeY, -1, OP_ID, 4, DG_FP_STR, OP_WRITE),
              op_arg_dat(nodeZ, -1, OP_ID, 4, DG_FP_STR, OP_WRITE));

  calc_mesh_constants();
}

void DGMesh3D::calc_mesh_constants() {
  const DG_FP *r_ptr = constants->get_mat_ptr(DGConstants::R) + (order_int - 1) * constants->Np_max;
  const DG_FP *s_ptr = constants->get_mat_ptr(DGConstants::S) + (order_int - 1) * constants->Np_max;
  const DG_FP *t_ptr = constants->get_mat_ptr(DGConstants::T) + (order_int - 1) * constants->Np_max;

  op_par_loop(init_grid_3d, "init_grid_3d", cells,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_gbl(r_ptr, DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(s_ptr, DG_NP, DG_FP_STR, OP_READ),
              op_arg_gbl(t_ptr, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(nodeX, -1, OP_ID, 4, DG_FP_STR, OP_READ),
              op_arg_dat(nodeY, -1, OP_ID, 4, DG_FP_STR, OP_READ),
              op_arg_dat(nodeZ, -1, OP_ID, 4, DG_FP_STR, OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(z, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::DR, x, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DS, x, 0.0, op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::DT, x, 0.0, op_tmp[2]);
  op_par_loop(init_geometric_factors_copy_3d, "init_geometric_factors_copy_3d", cells,
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[2], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rx, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
              op_arg_dat(sx, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
              op_arg_dat(tx, -1, OP_ID, 1, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::DR, y, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DS, y, 0.0, op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::DT, y, 0.0, op_tmp[2]);
  op_par_loop(init_geometric_factors_copy_3d, "init_geometric_factors_copy_3d", cells,
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[2], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ry, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
              op_arg_dat(sy, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
              op_arg_dat(ty, -1, OP_ID, 1, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::DR, z, 0.0, op_tmp[0]);
  op2_gemv(this, false, 1.0, DGConstants::DS, z, 0.0, op_tmp[1]);
  op2_gemv(this, false, 1.0, DGConstants::DT, z, 0.0, op_tmp[2]);
  op_par_loop(init_geometric_factors_copy_3d, "init_geometric_factors_copy_3d", cells,
              op_arg_dat(op_tmp[0], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[1], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op_tmp[2], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rz, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
              op_arg_dat(sz, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
              op_arg_dat(tz, -1, OP_ID, 1, DG_FP_STR, OP_WRITE));

  op_par_loop(init_geometric_factors_3d, "init_geometric_factors_3d", cells,
              op_arg_dat(rx, -1, OP_ID, 1, DG_FP_STR, OP_RW),
              op_arg_dat(ry, -1, OP_ID, 1, DG_FP_STR, OP_RW),
              op_arg_dat(rz, -1, OP_ID, 1, DG_FP_STR, OP_RW),
              op_arg_dat(sx, -1, OP_ID, 1, DG_FP_STR, OP_RW),
              op_arg_dat(sy, -1, OP_ID, 1, DG_FP_STR, OP_RW),
              op_arg_dat(sz, -1, OP_ID, 1, DG_FP_STR, OP_RW),
              op_arg_dat(tx, -1, OP_ID, 1, DG_FP_STR, OP_RW),
              op_arg_dat(ty, -1, OP_ID, 1, DG_FP_STR, OP_RW),
              op_arg_dat(tz, -1, OP_ID, 1, DG_FP_STR, OP_RW),
              op_arg_dat(J, -1, OP_ID, 1, DG_FP_STR, OP_WRITE));

  int num = 0;
  op_par_loop(face_check_3d, "face_check_3d", faces,
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
    std::cout << "Number of non matching points on faces: " << num << std::endl;
    exit(-1);
  }

  op_par_loop(init_faces_3d, "init_faces_3d", faces,
              op_arg_dat(faceNum, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(rx, -2, face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(ry, -2, face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(rz, -2, face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sx, -2, face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sy, -2, face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(sz, -2, face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tx, -2, face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(ty, -2, face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(tz, -2, face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(J,  -2, face2cells, 1, DG_FP_STR, OP_READ),
              op_arg_dat(nx, -1, OP_ID, 2, DG_FP_STR, OP_WRITE),
              op_arg_dat(ny, -1, OP_ID, 2, DG_FP_STR, OP_WRITE),
              op_arg_dat(nz, -1, OP_ID, 2, DG_FP_STR, OP_WRITE),
              op_arg_dat(sJ, -1, OP_ID, 2, DG_FP_STR, OP_WRITE),
              op_arg_dat(fscale, -1, OP_ID, 2, DG_FP_STR, OP_WRITE));

  int num_norm = 0;
  op_par_loop(normals_check_3d, "normals_check_3d", faces,
              op_arg_dat(order, -2, face2cells, 1, "int", OP_READ),
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
    std::cout << "Number of normal errors: " << num_norm << std::endl;
    exit(-1);
  }

  if(bface2cells) {
    op_par_loop(init_bfaces_3d, "init_bfaces_3d", bfaces,
                op_arg_dat(bfaceNum, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(rx, 0, bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(ry, 0, bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(rz, 0, bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(sx, 0, bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(sy, 0, bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(sz, 0, bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(tx, 0, bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(ty, 0, bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(tz, 0, bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(J,  0, bface2cells, 1, DG_FP_STR, OP_READ),
                op_arg_dat(bnx, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
                op_arg_dat(bny, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
                op_arg_dat(bnz, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
                op_arg_dat(bsJ, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
                op_arg_dat(bfscale, -1, OP_ID, 1, DG_FP_STR, OP_WRITE));
  }
}

void DGMesh3D::update_order(int new_order, std::vector<op_dat> &dats_to_interp) {
  for(int i = 0; i < dats_to_interp.size(); i++) {
    if(dats_to_interp[i]->dim != DG_NP) {
      std::cerr << "Interpolating between orders for non DG_NP dim dats is not implemented ...  exiting" << std::endl;
      exit(-1);
    }

    op_par_loop(interp_dat_to_new_order_3d, "interp_dat_to_new_order_3d", cells,
                op_arg_gbl(constants->get_mat_ptr(DGConstants::INTERP_MATRIX_ARRAY), DG_ORDER * DG_ORDER * DG_NP * DG_NP, DG_FP_STR, OP_READ),
                op_arg_gbl(&order_int, 1, "int", OP_READ),
                op_arg_gbl(&new_order, 1, "int", OP_READ),
                op_arg_dat(dats_to_interp[i], -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  }

  order_int = new_order;

  // Copy across new orders
  op_par_loop(copy_new_orders_int, "copy_new_orders_int", cells,
              op_arg_gbl(&new_order, 1, "int", OP_READ),
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_WRITE));

  calc_mesh_constants();
}
