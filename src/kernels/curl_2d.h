inline void curl_2d(const int *p, const DG_FP *div0, const DG_FP *div1,
                    const DG_FP *div2, const DG_FP *div3, const DG_FP *geof,
                    DG_FP *res) {
  // Get constants
  const int dg_np  = DG_CONSTANTS_TK[(*p - 1) * 5];

  const DG_FP rx = geof[RX_IND];
  const DG_FP sx = geof[SX_IND];
  const DG_FP ry = geof[RY_IND];
  const DG_FP sy = geof[SY_IND];
  for(int i = 0; i < dg_np; i++) {
    res[i] = rx * div2[i] + sx * div3[i] - ry * div0[i] - sy * div1[i];
  }
}
