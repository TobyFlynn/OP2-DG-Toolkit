inline void grad_2d(const int *p, const DG_FP *div0, const DG_FP *div1,
                    const DG_FP *rx, const DG_FP *sx, const DG_FP *ry,
                    const DG_FP *sy, DG_FP *ux, DG_FP *uy) {
  // Get constants
  const int dg_np  = DG_CONSTANTS[(*p - 1) * 5];

  for(int i = 0; i < dg_np; i++) {
    ux[i] = rx[i] * div0[i] + sx[i] * div1[i];
    uy[i] = ry[i] * div0[i] + sy[i] * div1[i];
  }
}
