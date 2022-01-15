#include "cublas_v2.h"

#include "op_seq.h"
#include "dg_blas_calls.h"
#include "dg_compiler_defs.h"

inline void cublas_init_gauss(const int numCells, const int *order,
                              const double *x_d, const double *y_d,
                              double *gxr_d, double *gxs_d, double *gyr_d,
                              double *gys_d) {
  double alpha = 1.0;
  double beta = 0.0;
  for(int c = 0; c < numCells; c++) {
    const double *x = x_d + c * DG_NP;
    const double *y = y_d + c * DG_NP;
    double *gxr = gxr_d + c * DG_G_NP;
    double *gxs = gxs_d + c * DG_G_NP;
    double *gyr = gyr_d + c * DG_G_NP;
    double *gys = gys_d + c * DG_G_NP;
    const int p = order[c];

    // Face 0
    cublasDgemv(handle, CUBLAS_OP_N, constants[p]->gNfp, constants[p]->Np, &alpha, constants[p]->gF0Dr_d, constants[p]->gNfp, x, 1, &beta, gxr, 1);
    cublasDgemv(handle, CUBLAS_OP_N, constants[p]->gNfp, constants[p]->Np, &alpha, constants[p]->gF0Ds_d, constants[p]->gNfp, x, 1, &beta, gxs, 1);
    cublasDgemv(handle, CUBLAS_OP_N, constants[p]->gNfp, constants[p]->Np, &alpha, constants[p]->gF0Dr_d, constants[p]->gNfp, y, 1, &beta, gyr, 1);
    cublasDgemv(handle, CUBLAS_OP_N, constants[p]->gNfp, constants[p]->Np, &alpha, constants[p]->gF0Ds_d, constants[p]->gNfp, y, 1, &beta, gys, 1);

    // Face 1
    cublasDgemv(handle, CUBLAS_OP_N, constants[p]->gNfp, constants[p]->Np, &alpha, constants[p]->gF1Dr_d, constants[p]->gNfp, x, 1, &beta, gxr + constants[p]->gNfp, 1);
    cublasDgemv(handle, CUBLAS_OP_N, constants[p]->gNfp, constants[p]->Np, &alpha, constants[p]->gF1Ds_d, constants[p]->gNfp, x, 1, &beta, gxs + constants[p]->gNfp, 1);
    cublasDgemv(handle, CUBLAS_OP_N, constants[p]->gNfp, constants[p]->Np, &alpha, constants[p]->gF1Dr_d, constants[p]->gNfp, y, 1, &beta, gyr + constants[p]->gNfp, 1);
    cublasDgemv(handle, CUBLAS_OP_N, constants[p]->gNfp, constants[p]->Np, &alpha, constants[p]->gF1Ds_d, constants[p]->gNfp, y, 1, &beta, gys + constants[p]->gNfp, 1);

    // Face 2
    cublasDgemv(handle, CUBLAS_OP_N, constants[p]->gNfp, constants[p]->Np, &alpha, constants[p]->gF2Dr_d, constants[p]->gNfp, x, 1, &beta, gxr + 2 * constants[p]->gNfp, 1);
    cublasDgemv(handle, CUBLAS_OP_N, constants[p]->gNfp, constants[p]->Np, &alpha, constants[p]->gF2Ds_d, constants[p]->gNfp, x, 1, &beta, gxs + 2 * constants[p]->gNfp, 1);
    cublasDgemv(handle, CUBLAS_OP_N, constants[p]->gNfp, constants[p]->Np, &alpha, constants[p]->gF2Dr_d, constants[p]->gNfp, y, 1, &beta, gyr + 2 * constants[p]->gNfp, 1);
    cublasDgemv(handle, CUBLAS_OP_N, constants[p]->gNfp, constants[p]->Np, &alpha, constants[p]->gF2Ds_d, constants[p]->gNfp, y, 1, &beta, gys + 2 * constants[p]->gNfp, 1);
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
  op_mpi_halo_exchanges_cuda(mesh->cells, 7, init_gauss_args);

  int setSize = mesh->x->set->size;
  int *tempOrder = (int *)malloc(setSize * sizeof(int));
  cudaMemcpy(tempOrder, mesh->order->data_d, setSize * sizeof(int), cudaMemcpyDeviceToHost);

  cublas_init_gauss(setSize, tempOrder, (double *)mesh->x->data_d,
                    (double *)mesh->y->data_d, (double *)gaussData->rx->data_d,
                    (double *)gaussData->sx->data_d,
                    (double *)gaussData->ry->data_d,
                    (double *)gaussData->sy->data_d);

  // Set correct dirty bits for OP2
  op_mpi_set_dirtybit_cuda(7, init_gauss_args);
}
