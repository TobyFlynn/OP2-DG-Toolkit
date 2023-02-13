inline void gemv_np_npT(const int *p, const DG_FP *alpha, const DG_FP *beta,
                        const DG_FP *matrix, const DG_FP *x, DG_FP *y) {
  // Get constants
  const int dg_np   = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS];
  const DG_FP *mat = &matrix[(*p - 1) * DG_NP * DG_NP];

  for(int i = 0; i < dg_np; i++) {
    y[i] *= *beta;
    for(int j = 0; j < dg_np; j++) {
      int ind = i * dg_np + j;
      y[i] += *alpha * mat[ind] * x[j];
    }
  }
}
