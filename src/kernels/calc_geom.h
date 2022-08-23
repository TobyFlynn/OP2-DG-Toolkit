inline void calc_geom(const int *p, const double *r_g, const double *s_g,
                      const double *nX, const double *nY,
                      double *x, double *y) {
  // Get constants for this element's order
  const int dg_np  = DG_CONSTANTS[(*p - 1) * 5];
  const double *r  = &r_g[(*p - 1) * DG_NP];
  const double *s  = &s_g[(*p - 1) * DG_NP];

  for(int i = 0; i < dg_np; i++) {
    x[i]  = 0.5 * nX[1] * (1.0 + r[i]);
    x[i] += 0.5 * nX[2] * (1.0 + s[i]);
    x[i] -= 0.5 * nX[0] * (s[i] + r[i]);

    y[i]  = 0.5 * nY[1] * (1.0 + r[i]);
    y[i] += 0.5 * nY[2] * (1.0 + s[i]);
    y[i] -= 0.5 * nY[0] * (s[i] + r[i]);
  }
}
