inline void J_3d(const int *p, const DG_FP *geof, const DG_FP *x, DG_FP *y) {
  // Get constants
  const int dg_np  = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  const DG_FP _J = geof[J_IND];
  for(int i = 0; i < dg_np; i++) {
    y[i] = _J * x[i];
  }
}
