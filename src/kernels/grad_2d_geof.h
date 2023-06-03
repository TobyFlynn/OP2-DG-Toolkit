inline void grad_2d_geof(const int *p, const DG_FP *geof, const DG_FP *tmp_r,
                         const DG_FP *tmp_s, DG_FP *ux, DG_FP *uy) {
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  const DG_FP rx = geof[RX_IND];
  const DG_FP ry = geof[RY_IND];
  const DG_FP sx = geof[SX_IND];
  const DG_FP sy = geof[SY_IND];
  for(int m = 0; m < dg_np; m++) {
    ux[m] = rx * tmp_r[m] + sx * tmp_s[m];
    uy[m] = ry * tmp_r[m] + sy * tmp_s[m];
  }
}
