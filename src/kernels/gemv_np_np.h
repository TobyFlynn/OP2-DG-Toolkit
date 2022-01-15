inline void gemv_np_np(const int *p, const double *alpha, const double *beta,
                       const double *matrix, const double *x, double *y) {
  // Get constants
  const int dg_np   = DG_CONSTANTS[(*p - 1) * 5];
  const double *mat = &matrix[(*p - 1) * DG_NP * DG_NP];

  for(int i = 0; i < dg_np; i++) {
    y[i] *= *beta;
    for(int j = 0; j < dg_np; j++) {
      int ind = i + j * dg_np;
      y[i] += *alpha * mat[ind] * x[j];
    }
  }
}
