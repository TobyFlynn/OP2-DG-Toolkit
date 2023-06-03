inline void div_2d_geof(const int *p, const DG_FP *geof, const DG_FP *u,
                         const DG_FP *v, DG_FP *tmp_r, DG_FP *tmp_s) {
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  const DG_FP rx = geof[RX_IND];
  const DG_FP ry = geof[RY_IND];
  const DG_FP sx = geof[SX_IND];
  const DG_FP sy = geof[SY_IND];
  for(int m = 0; m < dg_np; m++) {
    tmp_r[m] = rx * u[m] + ry * v[m];
    tmp_s[m] = sx * u[m] + sy * v[m];
  }
}
