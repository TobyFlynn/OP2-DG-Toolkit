inline void gemv_lift(const int *p, const double *alpha, const double *beta,
                      const double *matrix, const double *x, double *y) {
  // Get constants
  const int dg_np    = DG_CONSTANTS[(*p - 1) * 5];
  const int dg_nfp   = DG_CONSTANTS[(*p - 1) * 5 + 1];
  const double *lift = &matrix[(*p - 1) * 3 * DG_NPF * DG_NP];

  for(int i = 0; i < dg_np; i++) {
    y[i] *= *beta;
    for(int j = 0; j < 3 * dg_nfp; j++) {
      int ind = i + j * dg_np;
      y[i] += *alpha * lift[ind] * x[j];
    }
  }
}
