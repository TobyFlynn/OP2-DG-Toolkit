#include "op_seq.h"
#include "dg_blas_calls.h"
#include "dg_compiler_defs.h"

#define ARMA_ALLOW_FAKE_GCC
#include "armadillo"

void inv_blas(DGMesh *mesh, op_dat in, op_dat out) {
  op_arg inv_args[] = {
    op_arg_dat(in, -1, OP_ID, in->dim, "double", OP_READ),
    op_arg_dat(out, -1, OP_ID, out->dim, "double", OP_WRITE)
  };
  op_mpi_halo_exchanges(in->set, 2, inv_args);

  int setSize = in->set->size;

  for(int i = 0; i < setSize; i++) {
    const double *in_c = (double *)in->data + i * in->dim;
    double *inv_c      = (double *)out->data + i * out->dim;

    arma::mat a(in_c, DG_NP, DG_NP);
    arma::mat b(inv_c, DG_NP, DG_NP, false, true);

    b = arma::inv(a.t());
    b = b.t();
  }

  op_mpi_set_dirtybit(2, inv_args);
}