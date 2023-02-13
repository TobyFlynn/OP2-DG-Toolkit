inline void gemv_inv_mass_gauss_interpT(const int *p, const DG_FP *alpha,
                                        const DG_FP *beta, const DG_FP *matrix,
                                        const DG_FP *J, const DG_FP *x, DG_FP *y) {
  // Get constants
  const int dg_np   = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS];
  const int dg_g_np = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS + 3];
  const DG_FP *inv_mass_gauss_interp = &matrix[(*p - 1) * DG_G_NP * DG_NP];

  for(int i = 0; i < dg_np; i++) {
    DG_FP tmp = 0.0;
    for(int j = 0; j < dg_g_np; j++) {
      int ind = i + j * dg_np;
      tmp += *alpha * inv_mass_gauss_interp[ind] * x[j];
    }
    y[i] = *beta * y[i] + tmp / J[i];
  }
}
