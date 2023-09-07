inline void grad_over_int_2d_0(const DG_FP *geof, DG_FP *ux, DG_FP *uy) {
  const DG_FP rx = geof[RX_IND];
  const DG_FP ry = geof[RY_IND];
  const DG_FP sx = geof[SX_IND];
  const DG_FP sy = geof[SY_IND];
  
  for(int m = 0; m < DG_NP; m++) {
    const DG_FP tmp_r = ux[m];
    const DG_FP tmp_s = uy[m];
    ux[m] = rx * tmp_r + sx * tmp_s;
    uy[m] = ry * tmp_r + sy * tmp_s;
  }
}
