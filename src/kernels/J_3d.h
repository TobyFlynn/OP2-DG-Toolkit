inline void J_3d(const int *p, const DG_FP *geof, const DG_FP *x, DG_FP *y) {
  // Get constants
  const int dg_np  = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  for(int i = 0; i < dg_np; i++) {
    y[i] = geof[J_IND] * x[i];
  }
}
