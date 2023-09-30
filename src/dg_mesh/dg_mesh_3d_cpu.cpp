#include "dg_mesh/dg_mesh_3d.h"

#include "op_lib_cpp.h"

#include "dg_constants/dg_constants_3d.h"
#include "dg_compiler_defs.h"
#include "dg_op2_blas.h"
#include "dg_global_constants/dg_global_constants_3d.h"
#include "dg_dat_pool.h"

extern DGConstants *constants;

#ifdef USE_CUSTOM_MAPS

void DGMesh3D::free_custom_map(custom_map_info cmi) {
  if(cmi.map) {
    free(cmi.map);
  }
}

void DGMesh3D::update_custom_map() {
  if(node2node_custom_maps[order_int - 1].map) {
    node2node_custom_map = node2node_custom_maps[order_int - 1].map;
    node2node_custom_core_size = node2node_custom_maps[order_int - 1].core_size;
    node2node_custom_total_size = node2node_custom_maps[order_int - 1].total_size;
    return;
  }

  // order in args is just to force a halo exchange
  const int nargs = 4;
  op_arg args[] = {
    op_arg_dat(order, 0, face2cells, 1, "int", OP_RW),
    op_arg_dat(order, 1, face2cells, 1, "int", OP_RW),
    op_arg_dat(faceNum, -1, OP_ID, 2, "int", OP_READ),
    op_arg_dat(fmaskR, -1, OP_ID, DG_NPF, "int", OP_READ),
  };
  op_mpi_halo_exchanges(faces, nargs, args);

  const int dg_npf = DG_CONSTANTS_TK[(order_int - 1) * DG_NUM_CONSTANTS + 1];
  const int *fmask = &FMASK_TK[(order_int - 1) * DG_NUM_FACES * DG_NPF];
  const int *fmaskR_corrected_ptr = (int *)fmaskR->data;
  const int *faceNum_ptr = (int *)faceNum->data;

  std::vector<std::pair<std::pair<int,int>,std::pair<int,int>>> map_indices;
  for(int i = 0; i < faces->core_size; i++) {
    const int cell_indL = face2cells->map[2 * i];
    const int cell_indR = face2cells->map[2 * i + 1];
    const int faceNumL = faceNum_ptr[2 * i];
    const int faceNumR = faceNum_ptr[2 * i + 1];

    for(int j = 0; j < dg_npf; j++) {
      const int writeIndL = cell_indL * DG_NUM_FACES * DG_NPF + faceNumL * dg_npf + j;
      int writeIndR = 0;
      for(int n = 0; n < dg_npf; n++) {
        if(fmaskR_corrected_ptr[i * DG_NPF + j] == fmask[faceNumR * dg_npf + n])
          writeIndR = cell_indR * DG_NUM_FACES * DG_NPF + faceNumR * dg_npf + n;
      }
      const int readIndL = cell_indL * DG_NP + fmask[faceNumL * dg_npf + j];
      const int readIndR = cell_indR * DG_NP + fmaskR_corrected_ptr[i * DG_NPF + j];
      map_indices.push_back({{writeIndL, writeIndR}, {readIndL, readIndR}});
    }
  }

  node2node_custom_maps[order_int - 1].core_size = map_indices.size();
  std::sort(map_indices.begin(), map_indices.end());

  op_mpi_wait_all(nargs, args);

  for(int i = faces->core_size; i < faces->size + faces->exec_size; i++) {
    const int cell_indL = face2cells->map[2 * i];
    const int cell_indR = face2cells->map[2 * i + 1];
    const int faceNumL = faceNum_ptr[2 * i];
    const int faceNumR = faceNum_ptr[2 * i + 1];

    for(int j = 0; j < dg_npf; j++) {
      const int writeIndL = cell_indL * DG_NUM_FACES * DG_NPF + faceNumL * dg_npf + j;
      int writeIndR = 0;
      for(int n = 0; n < dg_npf; n++) {
        if(fmaskR_corrected_ptr[i * DG_NPF + j] == fmask[faceNumR * dg_npf + n])
          writeIndR = cell_indR * DG_NUM_FACES * DG_NPF + faceNumR * dg_npf + n;
      }
      const int readIndL = cell_indL * DG_NP + fmask[faceNumL * dg_npf + j];
      const int readIndR = cell_indR * DG_NP + fmaskR_corrected_ptr[i * DG_NPF + j];
      map_indices.push_back({{writeIndL, writeIndR}, {readIndL, readIndR}});
    }
  }

  op_mpi_set_dirtybit(nargs, args);

  node2node_custom_maps[order_int - 1].total_size = map_indices.size();
  if(faces->core_size < faces->size + faces->exec_size)
    std::sort(map_indices.begin() + node2node_custom_maps[order_int - 1].core_size, map_indices.end());

  node2node_custom_maps[order_int - 1].map = (int *)malloc(map_indices.size() * 4 * sizeof(int));
  for(int i = 0; i < map_indices.size(); i++) {
    node2node_custom_maps[order_int - 1].map[4 * i]     = map_indices[i].first.first;
    node2node_custom_maps[order_int - 1].map[4 * i + 1] = map_indices[i].first.second;
    node2node_custom_maps[order_int - 1].map[4 * i + 2] = map_indices[i].second.first;
    node2node_custom_maps[order_int - 1].map[4 * i + 3] = map_indices[i].second.second;
  }

  node2node_custom_map = node2node_custom_maps[order_int - 1].map;
  node2node_custom_core_size = node2node_custom_maps[order_int - 1].core_size;
  node2node_custom_total_size = node2node_custom_maps[order_int - 1].total_size;
}

#endif