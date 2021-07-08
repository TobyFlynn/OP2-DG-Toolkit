#include "op_seq.h"
#include "dg_blas_calls.h"

#define ARMA_ALLOW_FAKE_GCC
#include "armadillo"

void inv_mm_blas(DGMesh *mesh) {
  op_arg inv_args[] = {
    op_arg_dat(mesh->cubature->mm, -1, OP_ID, mesh->cubature->mm->dim, "double", OP_READ),
    op_arg_dat(mesh->cubature->mmInv, -1, OP_ID, mesh->cubature->mmInv->dim, "double", OP_WRITE)
  };
  op_mpi_halo_exchanges(mesh->cubature->mm->set, 2, inv_args);

  int setSize = mesh->cubature->mm->set->size;

  for(int i = 0; i < setSize; i++) {
    const double *mm_c = (double *)mesh->cubature->mm->data + i * mesh->cubature->mm->dim;
    double *mmInv_c = (double *)mesh->cubature->mmInv->data + i * mesh->cubature->mmInv->dim;

    arma::mat mm(mm_c, 15, 15);
    arma::mat mmInv(mmInv_c, 15, 15, false);

    mmInv = arma::inv(mm);
  }

  op_mpi_set_dirtybit(2, inv_args);
}
