#include "dg_mesh/dg_mesh_3d.h"

#include "dg_compiler_defs.h"

#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

__global__ void avg_3d_kerenel(
  const int *__restrict map,
  const double *__restrict in,
  double *__restrict out,
  int start,
  int end,
  int set_size) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid + start < end) {
    const int n = tid + start;
    const int writeIndL = map[n + set_size * 0];
    const int writeIndR = map[n + set_size * 1];
    const int readIndL  = map[n + set_size * 2];
    const int readIndR  = map[n + set_size * 3];
    const double avg = 0.5 * (in[readIndL] + in[readIndR]);
    out[writeIndL] = avg;
    out[writeIndR] = avg;
  }
}

void DGMesh3D::avg(op_dat in, op_dat out) {
  int nargs = 4;
  op_arg args[4];

  op_arg in_arg = op_arg_dat(in, -2, face2cells, DG_NP, DG_FP_STR, OP_READ);
  op_arg out_arg = op_arg_dat(out, -2, face2cells, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE);

  in_arg.idx = 0;
  args[0] = in_arg;
  for ( int v=1; v<2; v++ ){
    args[0 + v] = op_arg_dat(in_arg.dat, v, in_arg.map, DG_NP, DG_FP_STR, OP_READ);
  }

  out_arg.idx = 0;
  args[2] = out_arg;
  for ( int v=1; v<2; v++ ){
    args[2 + v] = op_arg_dat(out_arg.dat, v, out_arg.map, DG_NUM_FACES * DG_NPF, DG_FP_STR, OP_WRITE);
  }

  int ninds   = 2;
  int inds[4] = {0,0,1,1};

  if (OP_diags>2) {
    printf(" kernel routine with indirection: avg_3d\n");
  }
  int set_size = op_mpi_halo_exchanges_grouped(faces, nargs, args, 2);
  if (set_size > 0) {
    //set CUDA execution parameters
    int nthread = 64;

    for ( int round=0; round<2; round++ ){
      if (round==1) {
        op_mpi_wait_all_grouped(nargs, args, 2);
      }
      int start = round==0 ? 0 : node2node_custom_core_size;
      int end = round==0 ? node2node_custom_core_size : node2node_custom_total_size;
      if (end-start>0) {
        int nblocks = (end-start-1)/nthread+1;
        avg_3d_kerenel<<<nblocks,nthread>>>(
          node2node_custom_map_d,
          (double *)in_arg.data_d,
          (double *)out_arg.data_d,
          start,end,node2node_custom_total_size);
      }
    }
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
}
