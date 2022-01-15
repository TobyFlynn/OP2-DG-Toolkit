inline void curl(const int *p, const double *div0, const double *div1,
                 const double *div2, const double *div3, const double *rx,
                 const double *sx, const double *ry, const double *sy,
                 double *res) {
  // Get constants
  const int dg_np  = DG_CONSTANTS[(*p - 1) * 5];

  for(int i = 0; i < dg_np; i++) {
    res[i] = rx[i] * div2[i] + sx[i] * div3[i] - ry[i] * div0[i] - sy[i] * div1[i];
  }
}
