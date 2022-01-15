inline void cub_grad(const int *p, const double *rx, const double *sx,
                     const double *ry, const double *sy, const double *J,
                     double *temp0, double *temp1) {
  // Get constants for this element's order
  const int dg_cub_np = DG_CONSTANTS[(*p - 1) * 5 + 2];
  const double *cubW  = &cubW_g[(*p - 1) * DG_CUB_NP];

  for(int i = 0; i < dg_cub_np; i++) {
    double dru = temp0[i];
    double dsu = temp1[i];
    temp0[i] = cubW[i] * J[i] * (rx[i] * dru + sx[i] * dsu);
    temp1[i] = cubW[i] * J[i] * (ry[i] * dru + sy[i] * dsu);
  }
}
