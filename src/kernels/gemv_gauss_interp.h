inline void gemv_gauss_interp(const int *p, const bool *t, const double *alpha,
                              const double *beta, const double *matrix,
                              const double *x, double *y) {
  // Get constants
  const int dg_np   = DG_CONSTANTS[(*p - 1) * 5];
  const int dg_g_np = DG_CONSTANTS[(*p - 1) * 5 + 3];
  const double *gauss_interp = &matrix[(*p - 1) * DG_G_NP * DG_NP];

  if(*t) {
    for(int i = 0; i < dg_np; i++) {
      y[i] *= *beta;
      for(int j = 0; j < dg_g_np; j++) {
        int ind = i * dg_np + j;
        y[i] += *alpha * gauss_interp[ind] * x[j];
      }
    }
  } else {
    for(int i = 0; i < dg_g_np; i++) {
      y[i] *= *beta;
      for(int j = 0; j < dg_np; j++) {
        int ind = i + j * dg_g_np;
        y[i] += *alpha * gauss_interp[ind] * x[j];
      }
    }
  }
}
