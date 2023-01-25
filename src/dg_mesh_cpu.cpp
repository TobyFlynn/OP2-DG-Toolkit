#include "dg_mesh.h"

#include "dg_utils.h"

int DGMesh::get_local_vec_unknowns() {
  op_arg op2_args[] = {
    op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ)
  };
  op_mpi_halo_exchanges(order->set, 1, op2_args);
  const int setSize = order->set->size;
  const int *p = (int *)order->data;
  int local_unkowns = 0;
  for(int i = 0; i < setSize; i++) {
    int Np, Nfp;
    DGUtils::numNodes2D(p[i], &Np, &Nfp);
    local_unkowns += Np;
  }
  op_mpi_set_dirtybit(1, op2_args);
  return local_unkowns;
}
