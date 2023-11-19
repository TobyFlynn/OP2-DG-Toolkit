inline void fpmf_2d_grad_oi_sp(const int *p, const DG_FP *geof,
                         const DG_FP * __restrict__ fact,
                         float * __restrict__ ux, float * __restrict__ uy) {
  const float rx = (float)geof[RX_IND];
  const float sx = (float)geof[SX_IND];
  const float ry = (float)geof[RY_IND];
  const float sy = (float)geof[SY_IND];
  for(int m = 0; m < DG_CUB_2D_NP; m++) {
    const float r = ux[m];
    const float s = uy[m];
    const float factor = (float)fact[m];
    ux[m] = factor * (rx * r + sx * s);
    uy[m] = factor * (ry * r + sy * s);
  }
}
