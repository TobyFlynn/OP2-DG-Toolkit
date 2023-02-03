inline void cub_div_weak_2d(const int *p, double *temp0, double *temp1,
                            const double *rx, const double *sx, const double *ry,
                            const double *sy, const double *J, double *temp2,
                            double *temp3) {
  // Get constants for this element's order
  const int dg_cub_np = DG_CONSTANTS[(*p - 1) * 5 + 2];
  const double *cubW  = &cubW_g[(*p - 1) * DG_CUB_NP];

  for(int i = 0; i < dg_cub_np; i++) {
    double Vu = temp0[i];
    double Vv = temp1[i];
    temp0[i] = cubW[i] * J[i] * rx[i] * Vu;
    temp1[i] = cubW[i] * J[i] * sx[i] * Vu;
    temp2[i] = cubW[i] * J[i] * ry[i] * Vv;
    temp3[i] = cubW[i] * J[i] * sy[i] * Vv;
  }
}
