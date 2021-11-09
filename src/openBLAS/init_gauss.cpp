#include "cblas.h"

#include "op_seq.h"
#include "dg_blas_calls.h"
#include "dg_compiler_defs.h"

inline void openblas_init_gauss(const int numCells, const double *x,
                                const double *y, double *gxr, double *gxs,
                                double *gyr, double *gys) {
  double *dVMdr_f0 = (double *)malloc(DG_GF_NP * DG_NP * sizeof(double));
  double *dVMdr_f1 = (double *)malloc(DG_GF_NP * DG_NP * sizeof(double));
  double *dVMdr_f2 = (double *)malloc(DG_GF_NP * DG_NP * sizeof(double));
  double *dVMds_f0 = (double *)malloc(DG_GF_NP * DG_NP * sizeof(double));
  double *dVMds_f1 = (double *)malloc(DG_GF_NP * DG_NP * sizeof(double));
  double *dVMds_f2 = (double *)malloc(DG_GF_NP * DG_NP * sizeof(double));

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DG_GF_NP, DG_NP, DG_NP, 1.0, constants->gFInterp0, DG_GF_NP, constants->Dr, DG_NP, 0.0, dVMdr_f0, DG_GF_NP);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DG_GF_NP, DG_NP, DG_NP, 1.0, constants->gFInterp1, DG_GF_NP, constants->Dr, DG_NP, 0.0, dVMdr_f1, DG_GF_NP);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DG_GF_NP, DG_NP, DG_NP, 1.0, constants->gFInterp2, DG_GF_NP, constants->Dr, DG_NP, 0.0, dVMdr_f2, DG_GF_NP);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DG_GF_NP, DG_NP, DG_NP, 1.0, constants->gFInterp0, DG_GF_NP, constants->Ds, DG_NP, 0.0, dVMds_f0, DG_GF_NP);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DG_GF_NP, DG_NP, DG_NP, 1.0, constants->gFInterp1, DG_GF_NP, constants->Ds, DG_NP, 0.0, dVMds_f1, DG_GF_NP);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DG_GF_NP, DG_NP, DG_NP, 1.0, constants->gFInterp2, DG_GF_NP, constants->Ds, DG_NP, 0.0, dVMds_f2, DG_GF_NP);

  for(int c = 0; c < numCells; c++) {
    const double *x_c = x + c * DG_NP;
    const double *y_c = y + c * DG_NP;
    double *gxr_c = gxr + c * DG_G_NP;
    double *gxs_c = gxs + c * DG_G_NP;
    double *gyr_c = gyr + c * DG_G_NP;
    double *gys_c = gys + c * DG_G_NP;
    // Face 0
    cblas_dgemv(CblasColMajor, CblasNoTrans, DG_GF_NP, DG_NP, 1.0, dVMdr_f0, DG_GF_NP, x_c, 1, 0.0, gxr_c, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, DG_GF_NP, DG_NP, 1.0, dVMds_f0, DG_GF_NP, x_c, 1, 0.0, gxs_c, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, DG_GF_NP, DG_NP, 1.0, dVMdr_f0, DG_GF_NP, y_c, 1, 0.0, gyr_c, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, DG_GF_NP, DG_NP, 1.0, dVMds_f0, DG_GF_NP, y_c, 1, 0.0, gys_c, 1);

    // Face 1
    cblas_dgemv(CblasColMajor, CblasNoTrans, DG_GF_NP, DG_NP, 1.0, dVMdr_f1, DG_GF_NP, x_c, 1, 0.0, gxr_c + DG_GF_NP, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, DG_GF_NP, DG_NP, 1.0, dVMds_f1, DG_GF_NP, x_c, 1, 0.0, gxs_c + DG_GF_NP, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, DG_GF_NP, DG_NP, 1.0, dVMdr_f1, DG_GF_NP, y_c, 1, 0.0, gyr_c + DG_GF_NP, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, DG_GF_NP, DG_NP, 1.0, dVMds_f1, DG_GF_NP, y_c, 1, 0.0, gys_c + DG_GF_NP, 1);

    // Face 2
    cblas_dgemv(CblasColMajor, CblasNoTrans, DG_GF_NP, DG_NP, 1.0, dVMdr_f2, DG_GF_NP, x_c, 1, 0.0, gxr_c + 2 * DG_GF_NP, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, DG_GF_NP, DG_NP, 1.0, dVMds_f2, DG_GF_NP, x_c, 1, 0.0, gxs_c + 2 * DG_GF_NP, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, DG_GF_NP, DG_NP, 1.0, dVMdr_f2, DG_GF_NP, y_c, 1, 0.0, gyr_c + 2 * DG_GF_NP, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, DG_GF_NP, DG_NP, 1.0, dVMds_f2, DG_GF_NP, y_c, 1, 0.0, gys_c + 2 * DG_GF_NP, 1);
  }

  free(dVMdr_f0);
  free(dVMdr_f1);
  free(dVMdr_f2);
  free(dVMds_f0);
  free(dVMds_f1);
  free(dVMds_f2);
}

void init_gauss_blas(DGMesh *mesh, DGGaussData *gaussData) {
  // Make sure OP2 data is in the right place
  op_arg init_gauss_args[] = {
    op_arg_dat(mesh->x, -1, OP_ID, DG_NP, "double", OP_READ),
    op_arg_dat(mesh->y, -1, OP_ID, DG_NP, "double", OP_READ),
    op_arg_dat(gaussData->rx, -1, OP_ID, DG_G_NP, "double", OP_WRITE),
    op_arg_dat(gaussData->sx, -1, OP_ID, DG_G_NP, "double", OP_WRITE),
    op_arg_dat(gaussData->ry, -1, OP_ID, DG_G_NP, "double", OP_WRITE),
    op_arg_dat(gaussData->sy, -1, OP_ID, DG_G_NP, "double", OP_WRITE)
  };
  op_mpi_halo_exchanges(mesh->cells, 6, init_gauss_args);

  int setSize = mesh->x->set->size;

  openblas_init_gauss(setSize, (double *)mesh->x->data,
                     (double *)mesh->y->data, (double *)gaussData->rx->data,
                     (double *)gaussData->sx->data, (double *)gaussData->ry->data,
                     (double *)gaussData->sy->data);

  // Set correct dirty bits for OP2
  op_mpi_set_dirtybit(6, init_gauss_args);
}
