inline void fpmf_3d_grad_2(const int *p, const DG_FP *geof,
                           const DG_FP * __restrict__ fact,
                           DG_FP * __restrict__ ux, DG_FP * __restrict__ uy,
                           DG_FP *__restrict__ uz) {
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  const DG_FP rx = geof[RX_IND];
  const DG_FP sx = geof[SX_IND];
  const DG_FP tx = geof[TX_IND];
  const DG_FP ry = geof[RY_IND];
  const DG_FP sy = geof[SY_IND];
  const DG_FP ty = geof[TY_IND];
  const DG_FP rz = geof[RZ_IND];
  const DG_FP sz = geof[SZ_IND];
  const DG_FP tz = geof[TZ_IND];
  for(int m = 0; m < dg_np; m++) {
    const DG_FP r = ux[m];
    const DG_FP s = uy[m];
    const DG_FP t = uz[m];
    const DG_FP _fact = fact[m];
    ux[m] = _fact * (rx * r + sx * s + tx * t);
    uy[m] = _fact * (ry * r + sy * s + ty * t);
    uz[m] = _fact * (rz * r + sz * s + tz * t);
  }
}
