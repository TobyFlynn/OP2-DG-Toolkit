inline void cub_div_2d(const int *p, const double *rx, const double *sx,
                       const double *ry, const double *sy, const double *J,
                       double *temp0, const double *temp1, const double *temp2,
                       const double *temp3) {
  // Get constants for this element's order
  const int dg_cub_np = DG_CONSTANTS[(*p - 1) * 5 + 2];
  const double *cubW  = &cubW_g[(*p - 1) * DG_CUB_NP];

  for(int i = 0; i < dg_cub_np; i++) {
    double div = rx[i] * temp0[i] + sx[i] * temp1[i];
    div += ry[i] * temp2[i] + sy[i] * temp3[i];
    temp0[i] = cubW[i] * J[i] * div;
  }
}
