#include "cublas_v2.h"

#include "op_seq.h"
#include "dg_blas_calls.h"
#include "dg_compiler_defs.h"

inline void cublas_init_gauss(cublasHandle_t handle, const int numCells,
                              const double *x_d, const double *y_d, double *gxr_d,
                              double *gxs_d, double *gyr_d, double *gys_d) {
  double *dVMdr0_d;
  cudaMalloc((void**)&dVMdr0_d, DG_GF_NP * DG_NP * sizeof(double));
  double *dVMdr1_d;
  cudaMalloc((void**)&dVMdr1_d, DG_GF_NP * DG_NP * sizeof(double));
  double *dVMdr2_d;
  cudaMalloc((void**)&dVMdr2_d, DG_GF_NP * DG_NP * sizeof(double));

  double *dVMds0_d;
  cudaMalloc((void**)&dVMds0_d, DG_GF_NP * DG_NP * sizeof(double));
  double *dVMds1_d;
  cudaMalloc((void**)&dVMds1_d, DG_GF_NP * DG_NP * sizeof(double));
  double *dVMds2_d;
  cudaMalloc((void**)&dVMds2_d, DG_GF_NP * DG_NP * sizeof(double));

  double alpha = 1.0;
  double beta = 0.0;
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, DG_GF_NP, DG_NP, DG_NP, &alpha, constants->gFInterp0_d, DG_NP, constants->Dr_d, DG_NP, &beta, dVMdr0_d, DG_GF_NP);
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, DG_GF_NP, DG_NP, DG_NP, &alpha, constants->gFInterp1_d, DG_NP, constants->Dr_d, DG_NP, &beta, dVMdr1_d, DG_GF_NP);
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, DG_GF_NP, DG_NP, DG_NP, &alpha, constants->gFInterp2_d, DG_NP, constants->Dr_d, DG_NP, &beta, dVMdr2_d, DG_GF_NP);
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, DG_GF_NP, DG_NP, DG_NP, &alpha, constants->gFInterp0_d, DG_NP, constants->Ds_d, DG_NP, &beta, dVMds0_d, DG_GF_NP);
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, DG_GF_NP, DG_NP, DG_NP, &alpha, constants->gFInterp1_d, DG_NP, constants->Ds_d, DG_NP, &beta, dVMds1_d, DG_GF_NP);
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, DG_GF_NP, DG_NP, DG_NP, &alpha, constants->gFInterp2_d, DG_NP, constants->Ds_d, DG_NP, &beta, dVMds2_d, DG_GF_NP);

  for(int c = 0; c < numCells; c++) {
    const double *x = x_d + c * DG_NP;
    const double *y = y_d + c * DG_NP;
    double *gxr = gxr_d + c * DG_G_NP;
    double *gxs = gxs_d + c * DG_G_NP;
    double *gyr = gyr_d + c * DG_G_NP;
    double *gys = gys_d + c * DG_G_NP;

    // Face 0
    cublasDgemv(handle, CUBLAS_OP_N, DG_GF_NP, DG_NP, &alpha, dVMdr0_d, DG_GF_NP, x, 1, &beta, gxr, 1);
    cublasDgemv(handle, CUBLAS_OP_N, DG_GF_NP, DG_NP, &alpha, dVMds0_d, DG_GF_NP, x, 1, &beta, gxs, 1);
    cublasDgemv(handle, CUBLAS_OP_N, DG_GF_NP, DG_NP, &alpha, dVMdr0_d, DG_GF_NP, y, 1, &beta, gyr, 1);
    cublasDgemv(handle, CUBLAS_OP_N, DG_GF_NP, DG_NP, &alpha, dVMds0_d, DG_GF_NP, y, 1, &beta, gys, 1);

    // Face 1
    cublasDgemv(handle, CUBLAS_OP_N, DG_GF_NP, DG_NP, &alpha, dVMdr1_d, DG_GF_NP, x, 1, &beta, gxr + DG_GF_NP, 1);
    cublasDgemv(handle, CUBLAS_OP_N, DG_GF_NP, DG_NP, &alpha, dVMds1_d, DG_GF_NP, x, 1, &beta, gxs + DG_GF_NP, 1);
    cublasDgemv(handle, CUBLAS_OP_N, DG_GF_NP, DG_NP, &alpha, dVMdr1_d, DG_GF_NP, y, 1, &beta, gyr + DG_GF_NP, 1);
    cublasDgemv(handle, CUBLAS_OP_N, DG_GF_NP, DG_NP, &alpha, dVMds1_d, DG_GF_NP, y, 1, &beta, gys + DG_GF_NP, 1);

    // Face 2
    cublasDgemv(handle, CUBLAS_OP_N, DG_GF_NP, DG_NP, &alpha, dVMdr2_d, DG_GF_NP, x, 1, &beta, gxr + 2 * DG_GF_NP, 1);
    cublasDgemv(handle, CUBLAS_OP_N, DG_GF_NP, DG_NP, &alpha, dVMds2_d, DG_GF_NP, x, 1, &beta, gxs + 2 * DG_GF_NP, 1);
    cublasDgemv(handle, CUBLAS_OP_N, DG_GF_NP, DG_NP, &alpha, dVMdr2_d, DG_GF_NP, y, 1, &beta, gyr + 2 * DG_GF_NP, 1);
    cublasDgemv(handle, CUBLAS_OP_N, DG_GF_NP, DG_NP, &alpha, dVMds2_d, DG_GF_NP, y, 1, &beta, gys + 2 * DG_GF_NP, 1);
  }

  cudaFree(dVMdr0_d);
  cudaFree(dVMdr1_d);
  cudaFree(dVMdr2_d);
  cudaFree(dVMds0_d);
  cudaFree(dVMds1_d);
  cudaFree(dVMds2_d);
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
  op_mpi_halo_exchanges_cuda(mesh->cells, 6, init_gauss_args);

  int setSize = mesh->x->set->size;

  cublas_init_gauss(constants->handle, setSize, (double *)mesh->x->data_d,
                   (double *)mesh->y->data_d, (double *)gaussData->rx->data_d,
                   (double *)gaussData->sx->data_d, (double *)gaussData->ry->data_d,
                   (double *)gaussData->sy->data_d);

  // Set correct dirty bits for OP2
  op_mpi_set_dirtybit_cuda(6, init_gauss_args);
}