inline void J_3d_sp(const int *p, const DG_FP *geof, const float *x, float *y) {
  // Get constants
  const int dg_np  = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  const float tmp = (float)geof[J_IND];
  for(int i = 0; i < dg_np; i++) {
    y[i] = tmp * x[i];
  }
}
