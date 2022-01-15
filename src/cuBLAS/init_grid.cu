#include "cublas_v2.h"

#include "op_seq.h"
#include "dg_blas_calls.h"
#include "dg_compiler_defs.h"

inline void cublas_init_grid(const int numCells, const int *order,
                             const double *nodeX, const double *nodeY,
                             double *x_d, double *y_d, double *xr_d,
                             double *xs_d, double *yr_d, double *ys_d) {
  double *temp_d;
  cudaMalloc((void**)&temp_d, numCells * DG_NP * sizeof(double));

  for(int c = 0; c < numCells; c++) {
    // Get nodes for this cell (on host)
    const double n0[] = {nodeX[c * 3], nodeY[3 * c]};
    const double n1[] = {nodeX[c * 3 + 1], nodeY[3 * c + 1]};
    const double n2[] = {nodeX[c * 3 + 2], nodeY[3 * c + 2]};

    double *temp = temp_d + c * DG_NP;
    double *x  = x_d + c * DG_NP;
    double *y  = y_d + c * DG_NP;
    double *xr = xr_d + c * DG_NP;
    double *xs = xs_d + c * DG_NP;
    double *yr = yr_d + c * DG_NP;
    double *ys = ys_d + c * DG_NP;

    double alpha = 1.0;
    cublasDcopy(handle, constants[order[c]]->Np, constants[order[c]]->ones_d, 1, x, 1);
    cublasDaxpy(handle, constants[order[c]]->Np, &alpha, constants[order[c]]->r_d, 1, x, 1);
    alpha = 0.5 * n1[0];
    cublasDscal(handle, constants[order[c]]->Np, &alpha, x, 1);
    cublasDcopy(handle, constants[order[c]]->Np, constants[order[c]]->ones_d, 1, temp, 1);
    alpha = 1.0;
    cublasDaxpy(handle, constants[order[c]]->Np, &alpha, constants[order[c]]->s_d, 1, temp, 1);
    alpha = 0.5 * n2[0];
    cublasDaxpy(handle, constants[order[c]]->Np, &alpha, temp, 1, x, 1);
    cublasDcopy(handle, constants[order[c]]->Np, constants[order[c]]->s_d, 1, temp, 1);
    alpha = 1.0;
    cublasDaxpy(handle, constants[order[c]]->Np, &alpha, constants[order[c]]->r_d, 1, temp, 1);
    alpha = -0.5 * n0[0];
    cublasDaxpy(handle, constants[order[c]]->Np, &alpha, temp, 1, x, 1);

    cublasDcopy(handle, constants[order[c]]->Np, constants[order[c]]->ones_d, 1, y, 1);
    alpha = 1.0;
    cublasDaxpy(handle, constants[order[c]]->Np, &alpha, constants[order[c]]->r_d, 1, y, 1);
    alpha = 0.5 * n1[1];
    cublasDscal(handle, constants[order[c]]->Np, &alpha, y, 1);
    cublasDcopy(handle, constants[order[c]]->Np, constants[order[c]]->ones_d, 1, temp, 1);
    alpha = 1.0;
    cublasDaxpy(handle, constants[order[c]]->Np, &alpha, constants[order[c]]->s_d, 1, temp, 1);
    alpha = 0.5 * n2[1];
    cublasDaxpy(handle, constants[order[c]]->Np, &alpha, temp, 1, y, 1);
    cublasDcopy(handle, constants[order[c]]->Np, constants[order[c]]->s_d, 1, temp, 1);
    alpha = 1.0;
    cublasDaxpy(handle, constants[order[c]]->Np, &alpha, constants[order[c]]->r_d, 1, temp, 1);
    alpha = -0.5 * n0[1];
    cublasDaxpy(handle, constants[order[c]]->Np, &alpha, temp, 1, y, 1);

    double alpha2 = 1.0;
    double beta = 0.0;
    cublasDgemv(handle, CUBLAS_OP_N, constants[order[c]]->Np, constants[order[c]]->Np, &alpha2, constants[order[c]]->Dr_d, constants[order[c]]->Np, x, 1, &beta, xr, 1);
    cublasDgemv(handle, CUBLAS_OP_N, constants[order[c]]->Np, constants[order[c]]->Np, &alpha2, constants[order[c]]->Ds_d, constants[order[c]]->Np, x, 1, &beta, xs, 1);
    cublasDgemv(handle, CUBLAS_OP_N, constants[order[c]]->Np, constants[order[c]]->Np, &alpha2, constants[order[c]]->Dr_d, constants[order[c]]->Np, y, 1, &beta, yr, 1);
    cublasDgemv(handle, CUBLAS_OP_N, constants[order[c]]->Np, constants[order[c]]->Np, &alpha2, constants[order[c]]->Ds_d, constants[order[c]]->Np, y, 1, &beta, ys, 1);
  }

  cudaFree(temp_d);
}

void init_grid_blas(DGMesh *mesh) {
  // Make sure OP2 data is in the right place
  op_arg init_grid_args[] = {
    op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
    op_arg_dat(mesh->nodeX, -1, OP_ID, 3, "double", OP_READ),
    op_arg_dat(mesh->nodeY, -1, OP_ID, 3, "double", OP_READ),
    op_arg_dat(mesh->x, -1, OP_ID, DG_NP, "double", OP_WRITE),
    op_arg_dat(mesh->y, -1, OP_ID, DG_NP, "double", OP_WRITE),
    op_arg_dat(mesh->rx, -1, OP_ID, DG_NP, "double", OP_WRITE),
    op_arg_dat(mesh->sx, -1, OP_ID, DG_NP, "double", OP_WRITE),
    op_arg_dat(mesh->ry, -1, OP_ID, DG_NP, "double", OP_WRITE),
    op_arg_dat(mesh->sy, -1, OP_ID, DG_NP, "double", OP_WRITE)
  };
  op_mpi_halo_exchanges_cuda(mesh->cells, 9, init_grid_args);

  int setSize = mesh->x->set->size;
  int *tempOrder = (int *)malloc(setSize * sizeof(int));
  double *tempX  = (double *)malloc(setSize * 3 * sizeof(double));
  double *tempY  = (double *)malloc(setSize * 3 * sizeof(double));
  cudaMemcpy(tempOrder, mesh->order->data_d, setSize * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(tempX, mesh->nodeX->data_d, setSize * 3 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(tempY, mesh->nodeY->data_d, setSize * 3 * sizeof(double), cudaMemcpyDeviceToHost);

  cublas_init_grid(setSize, tempOrder, tempX, tempY, (double *)mesh->x->data_d,
                   (double *)mesh->y->data_d, (double *)mesh->rx->data_d,
                   (double *)mesh->sx->data_d, (double *)mesh->ry->data_d,
                   (double *)mesh->sy->data_d);

  free(tempX);
  free(tempY);

  // Set correct dirty bits for OP2
  op_mpi_set_dirtybit_cuda(9, init_grid_args);
}
