#include "cublas_v2.h"

#include "op_seq.h"
#include "dg_blas_calls.h"
#include "dg_compiler_defs.h"

inline void cublas_init_grid(cublasHandle_t handle, const int numCells,
                        const double *nodeX, const double *nodeY,
                        double *x_d, double *y_d, double *xr_d, double *xs_d,
                        double *yr_d, double *ys_d) {
  double *temp_d;
  cudaMalloc((void**)&temp_d, numCells * DG_NP * sizeof(double));

  for(int c = 0; c < numCells; c++) {
    // Get nodes for this cell (on host)
    const double n0[] = {nodeX[c * 3], nodeY[3 * c]};
    const double n1[] = {nodeX[c * 3 + 1], nodeY[3 * c + 1]};
    const double n2[] = {nodeX[c * 3 + 2], nodeY[3 * c + 2]};

    double *temp = temp_d + c * DG_NP;
    double *x = x_d + c * DG_NP;
    double *y = y_d + c * DG_NP;

    double alpha = 1.0;
    cublasDcopy(handle, DG_NP, constants->ones_d, 1, x, 1);
    cublasDaxpy(handle, DG_NP, &alpha, constants->r_d, 1, x, 1);
    alpha = 0.5 * n1[0];
    cublasDscal(handle, DG_NP, &alpha, x, 1);
    cublasDcopy(handle, DG_NP, constants->ones_d, 1, temp, 1);
    alpha = 1.0;
    cublasDaxpy(handle, DG_NP, &alpha, constants->s_d, 1, temp, 1);
    alpha = 0.5 * n2[0];
    cublasDaxpy(handle, DG_NP, &alpha, temp, 1, x, 1);
    cublasDcopy(handle, DG_NP, constants->s_d, 1, temp, 1);
    alpha = 1.0;
    cublasDaxpy(handle, DG_NP, &alpha, constants->r_d, 1, temp, 1);
    alpha = -0.5 * n0[0];
    cublasDaxpy(handle, DG_NP, &alpha, temp, 1, x, 1);

    cublasDcopy(handle, DG_NP, constants->ones_d, 1, y, 1);
    alpha = 1.0;
    cublasDaxpy(handle, DG_NP, &alpha, constants->r_d, 1, y, 1);
    alpha = 0.5 * n1[1];
    cublasDscal(handle, DG_NP, &alpha, y, 1);
    cublasDcopy(handle, DG_NP, constants->ones_d, 1, temp, 1);
    alpha = 1.0;
    cublasDaxpy(handle, DG_NP, &alpha, constants->s_d, 1, temp, 1);
    alpha = 0.5 * n2[1];
    cublasDaxpy(handle, DG_NP, &alpha, temp, 1, y, 1);
    cublasDcopy(handle, DG_NP, constants->s_d, 1, temp, 1);
    alpha = 1.0;
    cublasDaxpy(handle, DG_NP, &alpha, constants->r_d, 1, temp, 1);
    alpha = -0.5 * n0[1];
    cublasDaxpy(handle, DG_NP, &alpha, temp, 1, y, 1);
  }

  double alpha2 = 1.0;
  double beta = 0.0;
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, DG_NP, numCells, DG_NP, &alpha2, constants->Dr_d, DG_NP, x_d, DG_NP, &beta, xr_d, DG_NP);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, DG_NP, numCells, DG_NP, &alpha2, constants->Ds_d, DG_NP, x_d, DG_NP, &beta, xs_d, DG_NP);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, DG_NP, numCells, DG_NP, &alpha2, constants->Dr_d, DG_NP, y_d, DG_NP, &beta, yr_d, DG_NP);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, DG_NP, numCells, DG_NP, &alpha2, constants->Ds_d, DG_NP, y_d, DG_NP, &beta, ys_d, DG_NP);

  cudaFree(temp_d);
}

void init_grid_blas(DGMesh *mesh) {
  // Make sure OP2 data is in the right place
  op_arg init_grid_args[] = {
    op_arg_dat(mesh->nodeX, -1, OP_ID, 3, "double", OP_READ),
    op_arg_dat(mesh->nodeY, -1, OP_ID, 3, "double", OP_READ),
    op_arg_dat(mesh->x, -1, OP_ID, DG_NP, "double", OP_WRITE),
    op_arg_dat(mesh->y, -1, OP_ID, DG_NP, "double", OP_WRITE),
    op_arg_dat(mesh->rx, -1, OP_ID, DG_NP, "double", OP_WRITE),
    op_arg_dat(mesh->sx, -1, OP_ID, DG_NP, "double", OP_WRITE),
    op_arg_dat(mesh->ry, -1, OP_ID, DG_NP, "double", OP_WRITE),
    op_arg_dat(mesh->sy, -1, OP_ID, DG_NP, "double", OP_WRITE)
  };
  op_mpi_halo_exchanges_cuda(mesh->cells, 8, init_grid_args);

  int setSize = mesh->x->set->size;
  double *tempX = (double *)malloc(setSize * 3 * sizeof(double));
  double *tempY = (double *)malloc(setSize * 3 * sizeof(double));
  cudaMemcpy(tempX, mesh->nodeX->data_d, setSize * 3 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(tempY, mesh->nodeY->data_d, setSize * 3 * sizeof(double), cudaMemcpyDeviceToHost);

  cublas_init_grid(constants->handle, setSize, tempX, tempY, (double *)mesh->x->data_d,
                   (double *)mesh->y->data_d, (double *)mesh->rx->data_d,
                   (double *)mesh->sx->data_d, (double *)mesh->ry->data_d,
                   (double *)mesh->sy->data_d);

  free(tempX);
  free(tempY);

  // Set correct dirty bits for OP2
  op_mpi_set_dirtybit_cuda(8, init_grid_args);
}
