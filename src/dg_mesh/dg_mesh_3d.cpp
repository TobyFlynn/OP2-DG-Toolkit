#include "dg_mesh/dg_mesh_3d.h"

#include "op_seq.h"

#include <iostream>

#include "dg_constants/dg_constants_3d.h"
#include "dg_compiler_defs.h"
#include "dg_op2_blas.h"
#include "dg_global_constants/dg_global_constants_3d.h"
#include "dg_dat_pool.h"

#ifndef OP2_DG_CUDA
#ifdef OP2_DG_USE_LIBXSMM
#include "libxsmm_source.h"
#endif
#endif

DGConstants *constants;

DGDatPool3D *dg_dat_pool;

DGMesh3D::DGMesh3D(std::string &meshFile) {
  #ifndef OP2_DG_CUDA
  #ifdef OP2_DG_USE_LIBXSMM
  libxsmm_init();
  #endif
  #endif
  // Sets
  nodes   = op_decl_set_hdf5(meshFile.c_str(), "nodes");
  cells   = op_decl_set_hdf5(meshFile.c_str(), "cells");
  faces   = op_decl_set_hdf5(meshFile.c_str(), "faces");
  bfaces  = op_decl_set_hdf5(meshFile.c_str(), "bfaces");
  fluxes  = op_decl_set_hdf5(meshFile.c_str(), "fluxes");
  bfluxes = op_decl_set_hdf5(meshFile.c_str(), "bfluxes");

  // Maps
  cell2nodes  = op_decl_map_hdf5(cells, nodes, 4, meshFile.c_str(), "cell2nodes");
  face2nodes  = op_decl_map_hdf5(faces, nodes, 3, meshFile.c_str(), "face2nodes");
  face2cells  = op_decl_map_hdf5(faces, cells, 2, meshFile.c_str(), "face2cells");
  bface2nodes = op_decl_map_hdf5(bfaces, nodes, 3, meshFile.c_str(), "bface2nodes");
  bface2cells = op_decl_map_hdf5(bfaces, cells, 1, meshFile.c_str(), "bface2cells");
  flux2main_cell = op_decl_map_hdf5(fluxes, cells, 1, meshFile.c_str(), "flux2main_cell");
  flux2neighbour_cells = op_decl_map_hdf5(fluxes, cells, 4, meshFile.c_str(), "flux2neighbour_cells");
  flux2faces  = op_decl_map_hdf5(fluxes, faces, 4, meshFile.c_str(), "flux2faces");
  bflux2cells = op_decl_map_hdf5(bfluxes, cells, 2, meshFile.c_str(), "bflux2cells");
  bflux2faces = op_decl_map_hdf5(bfluxes, faces, 1, meshFile.c_str(), "bflux2faces");

  // Dats
  node_coords  = op_decl_dat_hdf5(nodes, 3, DG_FP_STR, meshFile.c_str(), "node_coords");
  faceNum      = op_decl_dat_hdf5(faces, 2, "int", meshFile.c_str(), "faceNum");
  bfaceNum     = op_decl_dat_hdf5(bfaces, 1, "int", meshFile.c_str(), "bfaceNum");
  periodicFace = op_decl_dat_hdf5(faces, 1, "int", meshFile.c_str(), "periodicFace");
  fluxL        = op_decl_dat_hdf5(fluxes, 4, "int", meshFile.c_str(), "fluxL");
  bfluxL       = op_decl_dat_hdf5(bfluxes, 1, "int", meshFile.c_str(), "bfluxL");

  nodeX = op_decl_dat(cells, 4, DG_FP_STR, (DG_FP *)NULL, "nodeX");
  nodeY = op_decl_dat(cells, 4, DG_FP_STR, (DG_FP *)NULL, "nodeY");
  nodeZ = op_decl_dat(cells, 4, DG_FP_STR, (DG_FP *)NULL, "nodeZ");

  x = op_decl_dat(cells, DG_NP, DG_FP_STR, (DG_FP *)NULL, "x");
  y = op_decl_dat(cells, DG_NP, DG_FP_STR, (DG_FP *)NULL, "y");
  z = op_decl_dat(cells, DG_NP, DG_FP_STR, (DG_FP *)NULL, "z");

  rx = op_decl_dat(cells, 1, DG_FP_STR, (DG_FP *)NULL, "rx");
  ry = op_decl_dat(cells, 1, DG_FP_STR, (DG_FP *)NULL, "ry");
  rz = op_decl_dat(cells, 1, DG_FP_STR, (DG_FP *)NULL, "rz");
  sx = op_decl_dat(cells, 1, DG_FP_STR, (DG_FP *)NULL, "sx");
  sy = op_decl_dat(cells, 1, DG_FP_STR, (DG_FP *)NULL, "sy");
  sz = op_decl_dat(cells, 1, DG_FP_STR, (DG_FP *)NULL, "sz");
  tx = op_decl_dat(cells, 1, DG_FP_STR, (DG_FP *)NULL, "tx");
  ty = op_decl_dat(cells, 1, DG_FP_STR, (DG_FP *)NULL, "ty");
  tz = op_decl_dat(cells, 1, DG_FP_STR, (DG_FP *)NULL, "tz");
  J  = op_decl_dat(cells, 1, DG_FP_STR, (DG_FP *)NULL, "J");
  geof = op_decl_dat(cells, 10, DG_FP_STR, (DG_FP *)NULL, "geof");

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

  order = op_decl_dat(cells, 1, "int", (int *)NULL, "order");

  fluxFaceNums = op_decl_dat(fluxes, 8, "int", (int *)NULL, "fluxFaceNums");
  fluxFmask = op_decl_dat(fluxes, 4 * DG_NPF, "int", (int *)NULL, "fluxFmask");

  fluxNx = op_decl_dat(fluxes, 4, DG_FP_STR, (DG_FP *)NULL, "fluxNx");
  fluxNy = op_decl_dat(fluxes, 4, DG_FP_STR, (DG_FP *)NULL, "fluxNy");
  fluxNz = op_decl_dat(fluxes, 4, DG_FP_STR, (DG_FP *)NULL, "fluxNz");
  fluxSJ = op_decl_dat(fluxes, 4, DG_FP_STR, (DG_FP *)NULL, "fluxSJ");
  fluxFscale = op_decl_dat(fluxes, 8, DG_FP_STR, (DG_FP *)NULL, "fluxFscale");

  dg_dat_pool = new DGDatPool3D(this);

  constants = new DGConstants3D(DG_ORDER);
  constants->calc_interp_mats();

  op_decl_const(DG_ORDER * DG_NPF * 4, "int", FMASK_TK);
  op_decl_const(DG_ORDER * 2, "int", DG_CONSTANTS_TK);

  order_int = DG_ORDER;
}


DGMesh3D::~DGMesh3D() {
  delete dg_dat_pool;
  delete constants;
  #ifndef OP2_DG_CUDA
  #ifdef OP2_DG_USE_LIBXSMM
  libxsmm_finalize();
  #endif
  #endif
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
  DGTempDat tmp0 = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp1 = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp2 = dg_dat_pool->requestTempDatCells(DG_NP);

  op_par_loop(init_grid_3d, "init_grid_3d", cells,
              op_arg_gbl(&order_int, 1, "int", OP_READ),
              op_arg_dat(nodeX, -1, OP_ID, 4, DG_FP_STR, OP_READ),
              op_arg_dat(nodeY, -1, OP_ID, 4, DG_FP_STR, OP_READ),
              op_arg_dat(nodeZ, -1, OP_ID, 4, DG_FP_STR, OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(y, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE),
              op_arg_dat(z, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::DR, x, 0.0, tmp0.dat);
  op2_gemv(this, false, 1.0, DGConstants::DS, x, 0.0, tmp1.dat);
  op2_gemv(this, false, 1.0, DGConstants::DT, x, 0.0, tmp2.dat);
  op_par_loop(init_geometric_factors_copy_3d, "init_geometric_factors_copy_3d", cells,
              op_arg_dat(tmp0.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp1.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp2.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rx, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
              op_arg_dat(sx, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
              op_arg_dat(tx, -1, OP_ID, 1, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::DR, y, 0.0, tmp0.dat);
  op2_gemv(this, false, 1.0, DGConstants::DS, y, 0.0, tmp1.dat);
  op2_gemv(this, false, 1.0, DGConstants::DT, y, 0.0, tmp2.dat);
  op_par_loop(init_geometric_factors_copy_3d, "init_geometric_factors_copy_3d", cells,
              op_arg_dat(tmp0.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp1.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp2.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(ry, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
              op_arg_dat(sy, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
              op_arg_dat(ty, -1, OP_ID, 1, DG_FP_STR, OP_WRITE));

  op2_gemv(this, false, 1.0, DGConstants::DR, z, 0.0, tmp0.dat);
  op2_gemv(this, false, 1.0, DGConstants::DS, z, 0.0, tmp1.dat);
  op2_gemv(this, false, 1.0, DGConstants::DT, z, 0.0, tmp2.dat);
  op_par_loop(init_geometric_factors_copy_3d, "init_geometric_factors_copy_3d", cells,
              op_arg_dat(tmp0.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp1.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp2.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(rz, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
              op_arg_dat(sz, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
              op_arg_dat(tz, -1, OP_ID, 1, DG_FP_STR, OP_WRITE));

  dg_dat_pool->releaseTempDatCells(tmp0);
  dg_dat_pool->releaseTempDatCells(tmp1);
  dg_dat_pool->releaseTempDatCells(tmp2);

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
              op_arg_dat(J, -1, OP_ID, 1, DG_FP_STR, OP_WRITE),
              op_arg_dat(geof, -1, OP_ID, 10, DG_FP_STR, OP_WRITE));

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

  op_par_loop(flux_init_3d, "flux_init_3d", fluxes,
              op_arg_dat(order, 0, flux2main_cell, 1, "int", OP_READ),
              op_arg_dat(faceNum, -4, flux2faces, 2, "int", OP_READ),
              op_arg_dat(fmaskL, -4, flux2faces, DG_NPF, "int", OP_READ),
              op_arg_dat(fmaskR, -4, flux2faces, DG_NPF, "int", OP_READ),
              op_arg_dat(nx, -4, flux2faces, 2, DG_FP_STR, OP_READ),
              op_arg_dat(ny, -4, flux2faces, 2, DG_FP_STR, OP_READ),
              op_arg_dat(nz, -4, flux2faces, 2, DG_FP_STR, OP_READ),
              op_arg_dat(sJ, -4, flux2faces, 2, DG_FP_STR, OP_READ),
              op_arg_dat(fscale, -4, flux2faces, 2, DG_FP_STR, OP_READ),
              op_arg_dat(fluxL, -1, OP_ID, 4, "int", OP_READ),
              op_arg_dat(fluxFaceNums, -1, OP_ID, 8, "int", OP_WRITE),
              op_arg_dat(fluxFmask, -1, OP_ID, 4 * DG_NPF, "int", OP_WRITE),
              op_arg_dat(fluxNx, -1, OP_ID, 4, DG_FP_STR, OP_WRITE),
              op_arg_dat(fluxNy, -1, OP_ID, 4, DG_FP_STR, OP_WRITE),
              op_arg_dat(fluxNz, -1, OP_ID, 4, DG_FP_STR, OP_WRITE),
              op_arg_dat(fluxSJ, -1, OP_ID, 4, DG_FP_STR, OP_WRITE),
              op_arg_dat(fluxFscale, -1, OP_ID, 8, DG_FP_STR, OP_WRITE));
}

void DGMesh3D::update_order(int new_order, std::vector<op_dat> &dats_to_interp) {
  for(int i = 0; i < dats_to_interp.size(); i++) {
    if(dats_to_interp[i]->dim != DG_NP) {
      std::cerr << "Interpolating between orders for non DG_NP dim dats is not implemented ...  exiting" << std::endl;
      exit(-1);
    }

    interp_dat_between_orders(order_int, new_order, dats_to_interp[i]);
  }

  order_int = new_order;

  // Copy across new orders
  op_par_loop(copy_new_orders_int, "copy_new_orders_int", cells,
              op_arg_gbl(&new_order, 1, "int", OP_READ),
              op_arg_dat(order, -1, OP_ID, 1, "int", OP_WRITE));

  calc_mesh_constants();
}
