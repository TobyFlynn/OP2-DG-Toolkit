inline void cub_grad_weak(const int *p, double *temp0, const double *rx,
                          const double *sx, const double *ry, const double *sy,
                          const double *J, double *temp1, double *temp2,
                          double *temp3) {
  // Get constants for this element's order
  const int dg_cub_np = DG_CONSTANTS[(*p - 1) * 5 + 2];
  const double *cubW  = &cubW_g[(*p - 1) * DG_CUB_NP];

  for(int i = 0; i < dg_cub_np; i++) {
    temp1[i] = cubW[i] * J[i] * sx[i] * temp0[i];
    temp2[i] = cubW[i] * J[i] * ry[i] * temp0[i];
    temp3[i] = cubW[i] * J[i] * sy[i] * temp0[i];
    temp0[i] = cubW[i] * J[i] * rx[i] * temp0[i];
  }
}
