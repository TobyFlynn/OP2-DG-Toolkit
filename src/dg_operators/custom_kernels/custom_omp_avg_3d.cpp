#include "dg_mesh/dg_mesh_3d.h"

#include "dg_compiler_defs.h"

#include "op_lib_cpp.h"

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
  int set_size = op_mpi_halo_exchanges(faces, nargs, args);
  if (set_size > 0) {
    const DG_FP *in_ptr = (DG_FP *)in_arg.data;
    DG_FP *out_ptr = (DG_FP *)out_arg.data;
    #pragma omp parallel for
    for(int n = 0; n < node2node_custom_core_size; n++){
      const int writeIndL = node2node_custom_map[n * 4];
      const int writeIndR = node2node_custom_map[n * 4 + 1];
      const int readIndL  = node2node_custom_map[n * 4 + 2];
      const int readIndR  = node2node_custom_map[n * 4 + 3];
      const double avg = 0.5 * (in_ptr[readIndL] + in_ptr[readIndR]);
      out_ptr[writeIndL] = avg;
      out_ptr[writeIndR] = avg;
    }

    op_mpi_wait_all(nargs, args);

    #pragma omp parallel for
    for(int n = node2node_custom_core_size; n < node2node_custom_total_size; n++){
      const int writeIndL = node2node_custom_map[n * 4];
      const int writeIndR = node2node_custom_map[n * 4 + 1];
      const int readIndL  = node2node_custom_map[n * 4 + 2];
      const int readIndR  = node2node_custom_map[n * 4 + 3];
      const double avg = 0.5 * (in_ptr[readIndL] + in_ptr[readIndR]);
      out_ptr[writeIndL] = avg;
      out_ptr[writeIndR] = avg;
    }
  }

  if (set_size == 0) {
    op_mpi_wait_all(nargs, args);
  }
  // combine reduction data
  op_mpi_set_dirtybit(nargs, args);
}

void DGMesh3D::avg_sp(op_dat in, op_dat out) {
  int nargs = 4;
  op_arg args[4];

  op_arg in_arg = op_arg_dat(in, -2, face2cells, DG_NP, "float", OP_READ);
  op_arg out_arg = op_arg_dat(out, -2, face2cells, DG_NUM_FACES * DG_NPF, "float", OP_WRITE);

  in_arg.idx = 0;
  args[0] = in_arg;
  for ( int v=1; v<2; v++ ){
    args[0 + v] = op_arg_dat(in_arg.dat, v, in_arg.map, DG_NP, "float", OP_READ);
  }

  out_arg.idx = 0;
  args[2] = out_arg;
  for ( int v=1; v<2; v++ ){
    args[2 + v] = op_arg_dat(out_arg.dat, v, out_arg.map, DG_NUM_FACES * DG_NPF, "float", OP_WRITE);
  }

  int ninds   = 2;
  int inds[4] = {0,0,1,1};

  if (OP_diags>2) {
    printf(" kernel routine with indirection: avg_3d\n");
  }
  int set_size = op_mpi_halo_exchanges(faces, nargs, args);
  if (set_size > 0) {
    const float *in_ptr = (float *)in_arg.data;
    float *out_ptr = (float *)out_arg.data;
    #pragma omp parallel for
    for(int n = 0; n < node2node_custom_core_size; n++){
      const int writeIndL = node2node_custom_map[n * 4];
      const int writeIndR = node2node_custom_map[n * 4 + 1];
      const int readIndL  = node2node_custom_map[n * 4 + 2];
      const int readIndR  = node2node_custom_map[n * 4 + 3];
      const float avg = 0.5 * (in_ptr[readIndL] + in_ptr[readIndR]);
      out_ptr[writeIndL] = avg;
      out_ptr[writeIndR] = avg;
    }

    op_mpi_wait_all(nargs, args);

    #pragma omp parallel for
    for(int n = node2node_custom_core_size; n < node2node_custom_total_size; n++){
      const int writeIndL = node2node_custom_map[n * 4];
      const int writeIndR = node2node_custom_map[n * 4 + 1];
      const int readIndL  = node2node_custom_map[n * 4 + 2];
      const int readIndR  = node2node_custom_map[n * 4 + 3];
      const float avg = 0.5 * (in_ptr[readIndL] + in_ptr[readIndR]);
      out_ptr[writeIndL] = avg;
      out_ptr[writeIndR] = avg;
    }
  }

  if (set_size == 0) {
    op_mpi_wait_all(nargs, args);
  }
  // combine reduction data
  op_mpi_set_dirtybit(nargs, args);
}
