#include "op_seq.h"
#include "dg_blas_calls.h"

#define ARMA_ALLOW_FAKE_GCC
#include "armadillo"

void inv_blas(DGMesh *mesh, op_dat in, op_dat out) {
  op_arg inv_args[] = {
    op_arg_dat(in, -1, OP_ID, in->dim, "double", OP_READ),
    op_arg_dat(out, -1, OP_ID, out->dim, "double", OP_WRITE)
  };
  op_mpi_halo_exchanges_cuda(in->set, 2, inv_args);

  int setSize = in->set->size;

  double *temp    = (double *)malloc(setSize * in->dim * sizeof(double));
  double *tempInv = (double *)malloc(setSize * out->dim * sizeof(double));
  cudaMemcpy(temp, in->data_d, setSize * in->dim * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(tempInv, out->data_d, setSize * out->dim * sizeof(double), cudaMemcpyDeviceToHost);

  for(int i = 0; i < setSize; i++) {
    const double *in_c = (double *)temp + i * in->dim;
    double *inv_c      = (double *)tempInv + i * out->dim;

    arma::mat a(in_c, 15, 15);
    arma::mat b(inv_c, 15, 15, false, true);

    b = arma::inv(a.t());
    b = b.t();
  }

  cudaMemcpy(out->data_d, tempInv, setSize * out->dim * sizeof(double), cudaMemcpyHostToDevice);

  free(temp);
  free(tempInv);

  op_mpi_set_dirtybit_cuda(2, inv_args);
}
