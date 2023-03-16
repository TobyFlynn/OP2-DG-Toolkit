inline void calc_geom(const int *p, const DG_FP *r_g, const DG_FP *s_g,
                      const DG_FP *nX, const DG_FP *nY,
                      DG_FP *x, DG_FP *y) {
  // Get constants for this element's order
  const int dg_np  = DG_CONSTANTS_TK[(*p - 1) * 5];
  const DG_FP *r  = &r_g[(*p - 1) * DG_NP];
  const DG_FP *s  = &s_g[(*p - 1) * DG_NP];

  for(int i = 0; i < dg_np; i++) {
    x[i]  = 0.5 * nX[1] * (1.0 + r[i]);
    x[i] += 0.5 * nX[2] * (1.0 + s[i]);
    x[i] -= 0.5 * nX[0] * (s[i] + r[i]);

    y[i]  = 0.5 * nY[1] * (1.0 + r[i]);
    y[i] += 0.5 * nY[2] * (1.0 + s[i]);
    y[i] -= 0.5 * nY[0] * (s[i] + r[i]);
  }
}
