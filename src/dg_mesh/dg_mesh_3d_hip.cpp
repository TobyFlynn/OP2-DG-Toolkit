#include "dg_mesh/dg_mesh_3d.h"

#include "op_lib_cpp.h"
// #include "op_cuda_rt_support.h"
// #include "op_cuda_reduction.h"

#include "dg_constants/dg_constants_3d.h"
#include "dg_compiler_defs.h"
#include "dg_op2_blas.h"
#include "dg_global_constants/dg_global_constants_3d.h"
#include "dg_dat_pool.h"

extern DGConstants *constants;

void DGMesh3D::free_custom_map(custom_map_info cmi) {
  throw std::runtime_error("DGMesh3D::free_custom_map not implemented for HIP yet");
  /*
  if(cmi.map) {
    free(cmi.map);
  }

  if(cmi.map_d) {
    cudaFree(cmi.map_d);
  }
  */
}

#ifdef DG_OP2_SOA
void DGMesh3D::update_custom_map() {
  throw std::runtime_error("DGMesh3D::update_custom_map not implemented for HIP yet");
  /*
  if(node2node_custom_maps[order_int - 1].map) {
    node2node_custom_map = node2node_custom_maps[order_int - 1].map;
    node2node_custom_map_d = node2node_custom_maps[order_int - 1].map_d;
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
  op_mpi_halo_exchanges_grouped(faces, nargs, args, 2, 0);
  op_mpi_wait_all_grouped(nargs, args, 2, 0);
  op_mpi_set_dirtybit_cuda(nargs, args);

  const int map_size = faces->size + faces->exec_size;
  const int dg_npf = DG_CONSTANTS_TK[(order_int - 1) * DG_NUM_CONSTANTS + 1];
  const int *fmask = &FMASK_TK[(order_int - 1) * DG_NUM_FACES * DG_NPF];
  op_arg fmaskR_args = op_arg_dat(fmaskR, -1, OP_ID, DG_NPF, "int", OP_READ);
  const int direct_stride = getSetSizeFromOpArg(&fmaskR_args);
  int *fmaskR_corrected_ptr = (int *)malloc(direct_stride * fmaskR->dim * sizeof(int));
  cudaMemcpy(fmaskR_corrected_ptr, fmaskR->data_d, direct_stride * fmaskR->dim * sizeof(int), cudaMemcpyDeviceToHost);
  int *faceNum_ptr = (int *)malloc(direct_stride * faceNum->dim * sizeof(int));
  cudaMemcpy(faceNum_ptr, faceNum->data_d, direct_stride * faceNum->dim * sizeof(int), cudaMemcpyDeviceToHost);
  int *face2cells_ptr = (int *)malloc(map_size * face2cells->dim * sizeof(int));
  cudaMemcpy(face2cells_ptr, face2cells->map_d, map_size * face2cells->dim * sizeof(int), cudaMemcpyDeviceToHost);

  op_arg order_args = op_arg_dat(order, -2, face2cells, 1, "int", OP_READ);
  const int indirect_stride = getSetSizeFromOpArg(&order_args);
  std::vector<std::pair<std::pair<int,int>,std::pair<int,int>>> map_indices;
  for(int i = 0; i < faces->core_size; i++) {
    const int cell_indL = face2cells_ptr[i];
    const int cell_indR = face2cells_ptr[i + map_size];
    const int faceNumL = faceNum_ptr[i];
    const int faceNumR = faceNum_ptr[i + direct_stride];

    for(int j = 0; j < dg_npf; j++) {
      // const int writeIndL = cell_indL * DG_NUM_FACES * DG_NPF + faceNumL * DG_NPF + j;
      const int writeIndL = cell_indL + (faceNumL * dg_npf + j) * indirect_stride;
      int writeIndR = 0;
      for(int n = 0; n < dg_npf; n++) {
        if(fmaskR_corrected_ptr[i + j * direct_stride] == fmask[faceNumR * dg_npf + n])
          writeIndR = cell_indR + (faceNumR * dg_npf + n) * indirect_stride;
      }
      const int readIndL = cell_indL + fmask[faceNumL * dg_npf + j] * indirect_stride;
      const int readIndR = cell_indR + fmaskR_corrected_ptr[i + j * direct_stride] * indirect_stride;
      map_indices.push_back({{writeIndL, writeIndR}, {readIndL, readIndR}});
    }
  }

  node2node_custom_maps[order_int - 1].core_size = map_indices.size();
  std::sort(map_indices.begin(), map_indices.end());

  for(int i = faces->core_size; i < faces->size + faces->exec_size; i++) {
    const int cell_indL = face2cells_ptr[i];
    const int cell_indR = face2cells_ptr[i + map_size];
    const int faceNumL = faceNum_ptr[i];
    const int faceNumR = faceNum_ptr[i + direct_stride];

    for(int j = 0; j < dg_npf; j++) {
      // const int writeIndL = cell_indL * DG_NUM_FACES * DG_NPF + faceNumL * DG_NPF + j;
      const int writeIndL = cell_indL + (faceNumL * dg_npf + j) * indirect_stride;
      int writeIndR = 0;
      for(int n = 0; n < dg_npf; n++) {
        if(fmaskR_corrected_ptr[i + j * direct_stride] == fmask[faceNumR * dg_npf + n])
          writeIndR = cell_indR + (faceNumR * dg_npf + n) * indirect_stride;
      }
      const int readIndL = cell_indL + fmask[faceNumL * dg_npf + j] * indirect_stride;
      const int readIndR = cell_indR + fmaskR_corrected_ptr[i + j * direct_stride] * indirect_stride;
      map_indices.push_back({{writeIndL, writeIndR}, {readIndL, readIndR}});
    }
  }

  node2node_custom_maps[order_int - 1].total_size = map_indices.size();
  if(faces->core_size < faces->size + faces->exec_size)
    std::sort(map_indices.begin() + node2node_custom_maps[order_int - 1].core_size, map_indices.end());

  node2node_custom_maps[order_int - 1].map = (int *)malloc(node2node_custom_maps[order_int - 1].total_size * 4 * sizeof(int));
  for(int i = 0; i < node2node_custom_maps[order_int - 1].total_size; i++) {
    node2node_custom_maps[order_int - 1].map[i] = map_indices[i].first.first;
    node2node_custom_maps[order_int - 1].map[i + node2node_custom_maps[order_int - 1].total_size] = map_indices[i].first.second;
    node2node_custom_maps[order_int - 1].map[i + 2 * node2node_custom_maps[order_int - 1].total_size] = map_indices[i].second.first;
    node2node_custom_maps[order_int - 1].map[i + 3 * node2node_custom_maps[order_int - 1].total_size] = map_indices[i].second.second;
  }

  cudaMalloc(&node2node_custom_maps[order_int - 1].map_d, node2node_custom_maps[order_int - 1].total_size * 4 * sizeof(int));
  cudaMemcpy(node2node_custom_maps[order_int - 1].map_d, node2node_custom_maps[order_int - 1].map, node2node_custom_maps[order_int - 1].total_size * 4 * sizeof(int), cudaMemcpyHostToDevice);

  free(fmaskR_corrected_ptr);
  free(faceNum_ptr);
  free(face2cells_ptr);

  node2node_custom_map = node2node_custom_maps[order_int - 1].map;
  node2node_custom_map_d = node2node_custom_maps[order_int - 1].map_d;
  node2node_custom_core_size = node2node_custom_maps[order_int - 1].core_size;
  node2node_custom_total_size = node2node_custom_maps[order_int - 1].total_size;
  */
}
#else
void DGMesh3D::update_custom_map() {
  throw std::runtime_error("DGMesh3D::update_custom_map not implemented for HIP yet");
  /*
  if(node2node_custom_maps[order_int - 1].map) {
    node2node_custom_map = node2node_custom_maps[order_int - 1].map;
    node2node_custom_map_d = node2node_custom_maps[order_int - 1].map_d;
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
  op_mpi_halo_exchanges_grouped(faces, nargs, args, 2, 0);
  op_mpi_wait_all_grouped(nargs, args, 2, 0);
  op_mpi_set_dirtybit_cuda(nargs, args);

  const int dg_npf = DG_CONSTANTS_TK[(order_int - 1) * DG_NUM_CONSTANTS + 1];
  const int *fmask = &FMASK_TK[(order_int - 1) * DG_NUM_FACES * DG_NPF];
  const int faces_size = faces->size + faces->exec_size;
  int *fmaskR_corrected_ptr = (int *)malloc(faces_size * fmaskR->dim * sizeof(int));
  cudaMemcpy(fmaskR_corrected_ptr, fmaskR->data_d, faces_size * fmaskR->dim * sizeof(int), cudaMemcpyDeviceToHost);
  int *faceNum_ptr = (int *)malloc(faces_size * faceNum->dim * sizeof(int));
  cudaMemcpy(faceNum_ptr, faceNum->data_d, faces_size * faceNum->dim * sizeof(int), cudaMemcpyDeviceToHost);
  int *face2cells_ptr = (int *)malloc(faces_size * face2cells->dim * sizeof(int));
  cudaMemcpy(face2cells_ptr, face2cells->map_d, faces_size * face2cells->dim * sizeof(int), cudaMemcpyDeviceToHost);

  std::vector<std::pair<std::pair<int,int>,std::pair<int,int>>> map_indices;
  for(int i = 0; i < faces->core_size; i++) {
    // const int cell_indL = face2cells_ptr[2 * i];
    // const int cell_indR = face2cells_ptr[2 * i + 1];
    const int cell_indL = face2cells_ptr[i];
    const int cell_indR = face2cells_ptr[i + faces_size];
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

  for(int i = faces->core_size; i < faces->size + faces->exec_size; i++) {
    // const int cell_indL = face2cells_ptr[2 * i];
    // const int cell_indR = face2cells_ptr[2 * i + 1];
    const int cell_indL = face2cells_ptr[i];
    const int cell_indR = face2cells_ptr[i + faces_size];
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

  node2node_custom_maps[order_int - 1].total_size = map_indices.size();
  if(faces->core_size < faces->size + faces->exec_size)
    std::sort(map_indices.begin() + node2node_custom_maps[order_int - 1].core_size, map_indices.end());

  node2node_custom_maps[order_int - 1].map = (int *)malloc(map_indices.size() * 4 * sizeof(int));
  // for(int i = 0; i < map_indices.size(); i++) {
  //   node2node_custom_maps[order_int - 1].map[4 * i]     = map_indices[i].first.first;
  //   node2node_custom_maps[order_int - 1].map[4 * i + 1] = map_indices[i].first.second;
  //   node2node_custom_maps[order_int - 1].map[4 * i + 2] = map_indices[i].second.first;
  //   node2node_custom_maps[order_int - 1].map[4 * i + 3] = map_indices[i].second.second;
  // }
  for(int i = 0; i < node2node_custom_maps[order_int - 1].total_size; i++) {
    node2node_custom_maps[order_int - 1].map[i] = map_indices[i].first.first;
    node2node_custom_maps[order_int - 1].map[i + node2node_custom_maps[order_int - 1].total_size] = map_indices[i].first.second;
    node2node_custom_maps[order_int - 1].map[i + 2 * node2node_custom_maps[order_int - 1].total_size] = map_indices[i].second.first;
    node2node_custom_maps[order_int - 1].map[i + 3 * node2node_custom_maps[order_int - 1].total_size] = map_indices[i].second.second;
  }

  cudaMalloc(&node2node_custom_maps[order_int - 1].map_d, map_indices.size() * 4 * sizeof(int));
  cudaMemcpy(node2node_custom_maps[order_int - 1].map_d, node2node_custom_maps[order_int - 1].map, map_indices.size() * 4 * sizeof(int), cudaMemcpyHostToDevice);

  free(fmaskR_corrected_ptr);
  free(faceNum_ptr);
  free(face2cells_ptr);

  node2node_custom_map = node2node_custom_maps[order_int - 1].map;
  node2node_custom_map_d = node2node_custom_maps[order_int - 1].map_d;
  node2node_custom_core_size = node2node_custom_maps[order_int - 1].core_size;
  node2node_custom_total_size = node2node_custom_maps[order_int - 1].total_size;
  */
}
#endif
