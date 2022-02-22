#include "op_seq.h"
#include "dg_blas_calls.h"
#include "dg_compiler_defs.h"
#include "dg_utils.h"

#define ARMA_ALLOW_FAKE_GCC
#include "armadillo"

cublasHandle_t handle;

void inv_blas(DGMesh *mesh, op_dat in, op_dat out) {
  op_arg inv_args[] = {
    op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
    op_arg_dat(in, -1, OP_ID, in->dim, "double", OP_READ),
    op_arg_dat(out, -1, OP_ID, out->dim, "double", OP_WRITE)
  };
  op_mpi_halo_exchanges_cuda(in->set, 3, inv_args);

  int setSize = in->set->size;

  int *tempOrder = (int *)malloc(setSize * sizeof(int));
  cudaMemcpy(tempOrder, mesh->order->data_d, setSize * sizeof(int), cudaMemcpyDeviceToHost);

  double *temp    = (double *)malloc(setSize * in->dim * sizeof(double));
  double *tempInv = (double *)malloc(setSize * out->dim * sizeof(double));
  cudaMemcpy(temp, in->data_d, setSize * in->dim * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(tempInv, out->data_d, setSize * out->dim * sizeof(double), cudaMemcpyDeviceToHost);

  for(int i = 0; i < setSize; i++) {
    const double *in_c = (double *)temp + i * in->dim;
    double *inv_c      = (double *)tempInv + i * out->dim;
    const int N        = tempOrder[i];
    int Np, Nfp;
    DGUtils::basic_constants(N, &Np, &Nfp);

    arma::mat a(in_c, Np, Np);
    arma::mat b(inv_c, Np, Np, false, true);

    b = arma::inv(a);
    // b = arma::inv_sympd(a);
  }

  cudaMemcpy(out->data_d, tempInv, setSize * out->dim * sizeof(double), cudaMemcpyHostToDevice);

  free(temp);
  free(tempInv);

  op_mpi_set_dirtybit_cuda(3, inv_args);
}

void init_blas() {
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
}

void destroy_blas() {
  cublasDestroy(handle);
}
