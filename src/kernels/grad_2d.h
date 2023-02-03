inline void grad_2d(const int *p, const double *div0, const double *div1,
                    const double *rx, const double *sx, const double *ry,
                    const double *sy, double *ux, double *uy) {
  // Get constants
  const int dg_np  = DG_CONSTANTS[(*p - 1) * 5];

  for(int i = 0; i < dg_np; i++) {
    ux[i] = rx[i] * div0[i] + sx[i] * div1[i];
    uy[i] = ry[i] * div0[i] + sy[i] * div1[i];
  }
}
