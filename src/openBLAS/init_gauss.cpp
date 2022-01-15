#include "cblas.h"

#include "op_seq.h"
#include "dg_blas_calls.h"
#include "dg_compiler_defs.h"

inline void openblas_init_gauss(const int numCells, const int *order, const double *x,
                                const double *y, double *gxr, double *gxs,
                                double *gyr, double *gys) {
  for(int c = 0; c < numCells; c++) {
    const double *x_c = x + c * DG_NP;
    const double *y_c = y + c * DG_NP;
    double *gxr_c = gxr + c * DG_G_NP;
    double *gxs_c = gxs + c * DG_G_NP;
    double *gyr_c = gyr + c * DG_G_NP;
    double *gys_c = gys + c * DG_G_NP;
    const int p   = order[c];
    // Face 0
    cblas_dgemv(CblasColMajor, CblasNoTrans, constants[p]->gNfp, constants[p]->Np, 1.0, constants[p]->gF0Dr, constants[p]->gNfp, x_c, 1, 0.0, gxr_c, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, constants[p]->gNfp, constants[p]->Np, 1.0, constants[p]->gF0Ds, constants[p]->gNfp, x_c, 1, 0.0, gxs_c, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, constants[p]->gNfp, constants[p]->Np, 1.0, constants[p]->gF0Dr, constants[p]->gNfp, y_c, 1, 0.0, gyr_c, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, constants[p]->gNfp, constants[p]->Np, 1.0, constants[p]->gF0Ds, constants[p]->gNfp, y_c, 1, 0.0, gys_c, 1);

    // Face 1
    cblas_dgemv(CblasColMajor, CblasNoTrans, constants[p]->gNfp, constants[p]->Np, 1.0, constants[p]->gF1Dr, constants[p]->gNfp, x_c, 1, 0.0, gxr_c + constants[p]->gNfp, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, constants[p]->gNfp, constants[p]->Np, 1.0, constants[p]->gF1Ds, constants[p]->gNfp, x_c, 1, 0.0, gxs_c + constants[p]->gNfp, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, constants[p]->gNfp, constants[p]->Np, 1.0, constants[p]->gF1Dr, constants[p]->gNfp, y_c, 1, 0.0, gyr_c + constants[p]->gNfp, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, constants[p]->gNfp, constants[p]->Np, 1.0, constants[p]->gF1Ds, constants[p]->gNfp, y_c, 1, 0.0, gys_c + constants[p]->gNfp, 1);

    // Face 2
    cblas_dgemv(CblasColMajor, CblasNoTrans, constants[p]->gNfp, constants[p]->Np, 1.0, constants[p]->gF2Dr, constants[p]->gNfp, x_c, 1, 0.0, gxr_c + 2 * constants[p]->gNfp, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, constants[p]->gNfp, constants[p]->Np, 1.0, constants[p]->gF2Ds, constants[p]->gNfp, x_c, 1, 0.0, gxs_c + 2 * constants[p]->gNfp, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, constants[p]->gNfp, constants[p]->Np, 1.0, constants[p]->gF2Dr, constants[p]->gNfp, y_c, 1, 0.0, gyr_c + 2 * constants[p]->gNfp, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, constants[p]->gNfp, constants[p]->Np, 1.0, constants[p]->gF2Ds, constants[p]->gNfp, y_c, 1, 0.0, gys_c + 2 * constants[p]->gNfp, 1);
  }
}

void init_gauss_blas(DGMesh *mesh, DGGaussData *gaussData) {
  // Make sure OP2 data is in the right place
  op_arg init_gauss_args[] = {
    op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
    op_arg_dat(mesh->x, -1, OP_ID, DG_NP, "double", OP_READ),
    op_arg_dat(mesh->y, -1, OP_ID, DG_NP, "double", OP_READ),
    op_arg_dat(gaussData->rx, -1, OP_ID, DG_G_NP, "double", OP_WRITE),
    op_arg_dat(gaussData->sx, -1, OP_ID, DG_G_NP, "double", OP_WRITE),
    op_arg_dat(gaussData->ry, -1, OP_ID, DG_G_NP, "double", OP_WRITE),
    op_arg_dat(gaussData->sy, -1, OP_ID, DG_G_NP, "double", OP_WRITE)
  };
  op_mpi_halo_exchanges(mesh->cells, 7, init_gauss_args);

  int setSize = mesh->x->set->size;

  openblas_init_gauss(setSize, (int *)mesh->order->data, (double *)mesh->x->data,
                     (double *)mesh->y->data, (double *)gaussData->rx->data,
                     (double *)gaussData->sx->data, (double *)gaussData->ry->data,
                     (double *)gaussData->sy->data);

  // Set correct dirty bits for OP2
  op_mpi_set_dirtybit(7, init_gauss_args);
}
