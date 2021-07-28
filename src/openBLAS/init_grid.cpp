#include "cblas.h"

#include "op_seq.h"
#include "dg_blas_calls.h"
#include "dg_compiler_defs.h"

inline void openblas_init_grid(const int numCells, const double *nodeX,
                               const double *nodeY, double *x, double *y,
                               double *xr, double *xs, double *yr, double *ys) {
  for(int c = 0; c < numCells; c++) {
    // Get nodes for this cell (on host)
    const double n0[] = {nodeX[c * 3], nodeY[3 * c]};
    const double n1[] = {nodeX[c * 3 + 1], nodeY[3 * c + 1]};
    const double n2[] = {nodeX[c * 3 + 2], nodeY[3 * c + 2]};

    double temp[DG_NP];
    double *x_c = x + c * DG_NP;
    double *y_c = y + c * DG_NP;
    double *xr_c = xr + c * DG_NP;
    double *xs_c = xs + c * DG_NP;
    double *yr_c = yr + c * DG_NP;
    double *ys_c = ys + c * DG_NP;

    cblas_dcopy(DG_NP, constants->ones, 1, x_c, 1);
    cblas_daxpy(DG_NP, 1.0, constants->r, 1, x_c, 1);
    cblas_dscal(DG_NP, 0.5 * n1[0], x_c, 1);
    cblas_dcopy(DG_NP, constants->ones, 1, temp, 1);
    cblas_daxpy(DG_NP, 1.0, constants->s, 1, temp, 1);
    cblas_daxpy(DG_NP, 0.5 * n2[0], temp, 1, x_c, 1);
    cblas_dcopy(DG_NP, constants->s, 1, temp, 1);
    cblas_daxpy(DG_NP, 1.0, constants->r, 1, temp, 1);
    cblas_daxpy(DG_NP, -0.5 * n0[0], temp, 1, x_c, 1);

    cblas_dcopy(DG_NP, constants->ones, 1, y_c, 1);
    cblas_daxpy(DG_NP, 1.0, constants->r, 1, y_c, 1);
    cblas_dscal(DG_NP, 0.5 * n1[1], y_c, 1);
    cblas_dcopy(DG_NP, constants->ones, 1, temp, 1);
    cblas_daxpy(DG_NP, 1.0, constants->s, 1, temp, 1);
    cblas_daxpy(DG_NP, 0.5 * n2[1], temp, 1, y_c, 1);
    cblas_dcopy(DG_NP, constants->s, 1, temp, 1);
    cblas_daxpy(DG_NP, 1.0, constants->r, 1, temp, 1);
    cblas_daxpy(DG_NP, -0.5 * n0[1], temp, 1, y_c, 1);
  }

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, DG_NP, numCells, DG_NP, 1.0, constants->Dr, DG_NP, x, DG_NP, 0.0, xr, DG_NP);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, DG_NP, numCells, DG_NP, 1.0, constants->Ds, DG_NP, x, DG_NP, 0.0, xs, DG_NP);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, DG_NP, numCells, DG_NP, 1.0, constants->Dr, DG_NP, y, DG_NP, 0.0, yr, DG_NP);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, DG_NP, numCells, DG_NP, 1.0, constants->Ds, DG_NP, y, DG_NP, 0.0, ys, DG_NP);
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
  op_mpi_halo_exchanges(mesh->cells, 8, init_grid_args);

  int setSize = mesh->x->set->size;

  openblas_init_grid(setSize, (double *)mesh->nodeX->data,
                     (double *)mesh->nodeY->data, (double *)mesh->x->data,
                     (double *)mesh->y->data, (double *)mesh->rx->data,
                     (double *)mesh->sx->data, (double *)mesh->ry->data,
                     (double *)mesh->sy->data);

  // Set correct dirty bits for OP2
  op_mpi_set_dirtybit(8, init_grid_args);
}