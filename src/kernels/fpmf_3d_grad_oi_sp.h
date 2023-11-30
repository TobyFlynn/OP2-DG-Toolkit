inline void fpmf_3d_grad_oi_sp(const DG_FP *geof, const DG_FP * __restrict__ fact,
                           float * __restrict__ ux, float * __restrict__ uy,
                           float *__restrict__ uz) {
  // const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  const float rx = (float)geof[RX_IND];
  const float sx = (float)geof[SX_IND];
  const float tx = (float)geof[TX_IND];
  const float ry = (float)geof[RY_IND];
  const float sy = (float)geof[SY_IND];
  const float ty = (float)geof[TY_IND];
  const float rz = (float)geof[RZ_IND];
  const float sz = (float)geof[SZ_IND];
  const float tz = (float)geof[TZ_IND];
  for(int m = 0; m < DG_CUB_3D_NP; m++) {
    const float r = ux[m];
    const float s = uy[m];
    const float t = uz[m];
    const float _fact = (float)fact[m];
    ux[m] = _fact * (rx * r + sx * s + tx * t);
    uy[m] = _fact * (ry * r + sy * s + ty * t);
    uz[m] = _fact * (rz * r + sz * s + tz * t);
  }
}
