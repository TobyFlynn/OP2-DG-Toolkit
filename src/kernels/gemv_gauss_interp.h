inline void gemv_gauss_interp(const int *p, const DG_FP *alpha,
                              const DG_FP *beta, const DG_FP *matrix,
                              const DG_FP *x, DG_FP *y) {
  // Get constants
  const int dg_np   = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS];
  const int dg_g_np = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS + 3];
  const DG_FP *gauss_interp = &matrix[(*p - 1) * DG_G_NP * DG_NP];

  for(int i = 0; i < dg_g_np; i++) {
    y[i] *= *beta;
    for(int j = 0; j < dg_np; j++) {
      int ind = i + j * dg_g_np;
      y[i] += *alpha * gauss_interp[ind] * x[j];
    }
  }
}
