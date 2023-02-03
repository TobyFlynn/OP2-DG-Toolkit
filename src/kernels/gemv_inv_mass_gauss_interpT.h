inline void gemv_inv_mass_gauss_interpT(const int *p, const double *alpha,
                                        const double *beta, const double *matrix,
                                        const double *J, const double *x, double *y) {
  // Get constants
  const int dg_np   = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS];
  const int dg_g_np = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS + 3];
  const double *inv_mass_gauss_interp = &matrix[(*p - 1) * DG_G_NP * DG_NP];

  for(int i = 0; i < dg_np; i++) {
    double tmp = 0.0;
    for(int j = 0; j < dg_g_np; j++) {
      int ind = i + j * dg_np;
      tmp += *alpha * inv_mass_gauss_interp[ind] * x[j];
    }
    y[i] = *beta * y[i] + tmp / J[i];
  }
}
