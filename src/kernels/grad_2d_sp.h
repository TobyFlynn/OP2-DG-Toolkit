inline void grad_2d_sp(const int *p, const DG_FP *geof, const float *ur,
                       const float *us, float *ux, float *uy) {
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  const float rx = (float)geof[RX_IND];
  const float sx = (float)geof[SX_IND];
  const float ry = (float)geof[RY_IND];
  const float sy = (float)geof[SY_IND];
  for(int m = 0; m < dg_np; m++) {
    ux[m] = rx * ur[m] + sx * us[m];
    uy[m] = ry * ur[m] + sy * us[m];
  }
}
