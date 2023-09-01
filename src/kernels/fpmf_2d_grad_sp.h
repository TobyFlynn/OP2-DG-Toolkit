inline void fpmf_2d_grad_sp(const int *p, const DG_FP *geof,
                            const DG_FP * __restrict__ fact,
                            float * __restrict__ ux, float * __restrict__ uy) {
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  const float _rx = (float)geof[RX_IND];
  const float _sx = (float)geof[SX_IND];
  const float _ry = (float)geof[RY_IND];
  const float _sy = (float)geof[SY_IND];
  for(int m = 0; m < dg_np; m++) {
    const float r = ux[m];
    const float s = uy[m];
    const float _fact = (float)fact[m];
    ux[m] = _fact * (_rx * r + _sx * s);
    uy[m] = _fact * (_ry * r + _sy * s);
  }
}
