#include "dg_mesh.h"

#include "dg_utils.h"

int DGMesh::get_local_vec_unknowns() {
  op_arg op2_args[] = {
    op_arg_dat(order, -1, OP_ID, 1, "int", OP_READ)
  };
  op_mpi_halo_exchanges_cuda(order->set, 1, op2_args);
  const int setSize = order->set->size;
  int *tempOrder = (int *)malloc(setSize * sizeof(int));
  cudaMemcpy(tempOrder, order->data_d, setSize * sizeof(int), cudaMemcpyDeviceToHost);
  int local_unkowns = 0;
  for(int i = 0; i < setSize; i++) {
    int Np, Nfp;
    DGUtils::numNodes2D(tempOrder[i], &Np, &Nfp);
    local_unkowns += Np;
  }
  free(tempOrder);
  op_mpi_set_dirtybit_cuda(1, op2_args);
  return local_unkowns;
}
