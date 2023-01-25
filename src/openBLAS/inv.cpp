#include "op_seq.h"
#include "dg_blas_calls.h"
#include "dg_compiler_defs.h"
#include "dg_utils.h"

#define ARMA_ALLOW_FAKE_GCC
#include "armadillo"

void inv_blas(DGMesh *mesh, op_dat in, op_dat out) {
  op_arg inv_args[] = {
    op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
    op_arg_dat(in, -1, OP_ID, in->dim, "double", OP_READ),
    op_arg_dat(out, -1, OP_ID, out->dim, "double", OP_WRITE)
  };
  op_mpi_halo_exchanges(in->set, 3, inv_args);

  int setSize = in->set->size;

  const int *p = (int *)mesh->order->data;

  for(int i = 0; i < setSize; i++) {
    const double *in_c = (double *)in->data + i * in->dim;
    double *inv_c      = (double *)out->data + i * out->dim;
    const int N        = p[i];
    int Np, Nfp;
    DGUtils::numNodes2D(N, &Np, &Nfp);

    arma::mat a(in_c, Np, Np);
    arma::mat b(inv_c, Np, Np, false, true);

    b = arma::inv(a);
    // b = arma::inv_sympd(a);
  }

  op_mpi_set_dirtybit(3, inv_args);
}

void init_blas() {

}

void destroy_blas() {

}
