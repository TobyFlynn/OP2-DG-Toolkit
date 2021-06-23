#include "cblas.h"

#include "op_seq.h"
#include "dg_blas_calls.h"

inline void openblas_init_gauss(const int numCells, const double *x,
                                const double *y, double *gxr, double *gxs,
                                double *gyr, double *gys) {
  double *dVMdr_f0 = (double *)malloc(7 * 15 * sizeof(double));
  double *dVMdr_f1 = (double *)malloc(7 * 15 * sizeof(double));
  double *dVMdr_f2 = (double *)malloc(7 * 15 * sizeof(double));
  double *dVMds_f0 = (double *)malloc(7 * 15 * sizeof(double));
  double *dVMds_f1 = (double *)malloc(7 * 15 * sizeof(double));
  double *dVMds_f2 = (double *)malloc(7 * 15 * sizeof(double));

  cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, 7, 15, 15, 1.0, constants->gFInterp0, 15, constants->Dr, 15, 0.0, dVMdr_f0, 7);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, 7, 15, 15, 1.0, constants->gFInterp1, 15, constants->Dr, 15, 0.0, dVMdr_f1, 7);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, 7, 15, 15, 1.0, constants->gFInterp2, 15, constants->Dr, 15, 0.0, dVMdr_f2, 7);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, 7, 15, 15, 1.0, constants->gFInterp0, 15, constants->Ds, 15, 0.0, dVMds_f0, 7);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, 7, 15, 15, 1.0, constants->gFInterp1, 15, constants->Ds, 15, 0.0, dVMds_f1, 7);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, 7, 15, 15, 1.0, constants->gFInterp2, 15, constants->Ds, 15, 0.0, dVMds_f2, 7);

  for(int c = 0; c < numCells; c++) {
    const double *x_c = x + c * 15;
    const double *y_c = y + c * 15;
    double *gxr_c = gxr + c * 21;
    double *gxs_c = gxs + c * 21;
    double *gyr_c = gyr + c * 21;
    double *gys_c = gys + c * 21;
    // Face 0
    cblas_dgemv(CblasColMajor, CblasNoTrans, 7, 15, 1.0, dVMdr_f0, 7, x_c, 1, 0.0, gxr_c, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, 7, 15, 1.0, dVMds_f0, 7, x_c, 1, 0.0, gxs_c, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, 7, 15, 1.0, dVMdr_f0, 7, y_c, 1, 0.0, gyr_c, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, 7, 15, 1.0, dVMds_f0, 7, y_c, 1, 0.0, gys_c, 1);

    // Face 1
    cblas_dgemv(CblasColMajor, CblasNoTrans, 7, 15, 1.0, dVMdr_f1, 7, x_c, 1, 0.0, gxr_c + 7, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, 7, 15, 1.0, dVMds_f1, 7, x_c, 1, 0.0, gxs_c + 7, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, 7, 15, 1.0, dVMdr_f1, 7, y_c, 1, 0.0, gyr_c + 7, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, 7, 15, 1.0, dVMds_f1, 7, y_c, 1, 0.0, gys_c + 7, 1);

    // Face 2
    cblas_dgemv(CblasColMajor, CblasNoTrans, 7, 15, 1.0, dVMdr_f2, 7, x_c, 1, 0.0, gxr_c + 14, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, 7, 15, 1.0, dVMds_f2, 7, x_c, 1, 0.0, gxs_c + 14, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, 7, 15, 1.0, dVMdr_f2, 7, y_c, 1, 0.0, gyr_c + 14, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, 7, 15, 1.0, dVMds_f2, 7, y_c, 1, 0.0, gys_c + 14, 1);
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
    op_arg_dat(mesh->x, -1, OP_ID, 15, "double", OP_READ),
    op_arg_dat(mesh->y, -1, OP_ID, 15, "double", OP_READ),
    op_arg_dat(gaussData->rx, -1, OP_ID, 21, "double", OP_WRITE),
    op_arg_dat(gaussData->sx, -1, OP_ID, 21, "double", OP_WRITE),
    op_arg_dat(gaussData->ry, -1, OP_ID, 21, "double", OP_WRITE),
    op_arg_dat(gaussData->sy, -1, OP_ID, 21, "double", OP_WRITE)
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
