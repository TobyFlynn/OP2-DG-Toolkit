inline void gemv_cub_np_npT(const int *p, const double *alpha,
                            const double *beta, const double *matrix,
                            const double *x, double *y) {
  // Get constants
  const int dg_np     = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS];
  const int dg_cub_np = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS + 2];
  const double *mat   = &matrix[(*p - 1) * DG_CUB_NP * DG_NP];

  for(int i = 0; i < dg_np; i++) {
    y[i] *= *beta;
    for(int j = 0; j < dg_cub_np; j++) {
      int ind = i * dg_cub_np + j;
      y[i] += *alpha * mat[ind] * x[j];
    }
  }
}
