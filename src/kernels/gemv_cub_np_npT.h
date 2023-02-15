inline void gemv_cub_np_npT(const int *p, const DG_FP *alpha,
                            const DG_FP *beta, const DG_FP *matrix,
                            const DG_FP *x, DG_FP *y) {
  // Get constants
  const int dg_np     = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS];
  const int dg_cub_np = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS + 2];
  const DG_FP *mat   = &matrix[(*p - 1) * DG_CUB_NP * DG_NP];

  for(int i = 0; i < dg_np; i++) {
    y[i] *= *beta;
    for(int j = 0; j < dg_cub_np; j++) {
      // int ind = i * dg_cub_np + j;
      int ind = DG_MAT_IND(j, i, dg_cub_np, dg_np);
      y[i] += *alpha * mat[ind] * x[j];
    }
  }
}
