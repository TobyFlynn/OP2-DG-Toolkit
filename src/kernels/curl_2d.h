inline void curl_2d(const int *p, const DG_FP *div0, const DG_FP *div1,
                    const DG_FP *div2, const DG_FP *div3, const DG_FP *rx,
                    const DG_FP *sx, const DG_FP *ry, const DG_FP *sy,
                    DG_FP *res) {
  // Get constants
  const int dg_np  = DG_CONSTANTS_TK[(*p - 1) * 5];

  for(int i = 0; i < dg_np; i++) {
    res[i] = rx[i] * div2[i] + sx[i] * div3[i] - ry[i] * div0[i] - sy[i] * div1[i];
  }
}
