inline void fpmf_3d_grad_2(const int * __restrict__ p, const DG_FP * __restrict__ geof,
                           const DG_FP * __restrict__ fact,
                           DG_FP * __restrict__ ux, DG_FP * __restrict__ uy,
                           DG_FP *__restrict__ uz) {
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  const DG_FP _rx = geof[RX_IND];
  const DG_FP _ry = geof[RY_IND];
  const DG_FP _rz = geof[RZ_IND];
  const DG_FP _sx = geof[SX_IND];
  const DG_FP _sy = geof[SY_IND];
  const DG_FP _sz = geof[SZ_IND];
  const DG_FP _tx = geof[TX_IND];
  const DG_FP _ty = geof[TY_IND];
  const DG_FP _tz = geof[TZ_IND];
  for(int m = 0; m < dg_np; m++) {
    const DG_FP r = ux[m];
    const DG_FP s = uy[m];
    const DG_FP t = uz[m];
    const DG_FP _fact = fact[m];
    ux[m] = _fact * (_rx * r + _sx * s + _tx * t);
    uy[m] = _fact * (_ry * r + _sy * s + _ty * t);
    uz[m] = _fact * (_rz * r + _sz * s + _tz * t);
  }
}
