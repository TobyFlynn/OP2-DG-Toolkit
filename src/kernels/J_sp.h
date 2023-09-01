inline void J_sp(const int *p, const DG_FP *geof, const float *tmp, float *u) {
  // Get constants for this element's order
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * 5];
  const float _J = (float)geof[J_IND];
  for(int i = 0; i < dg_np; i++) {
    u[i] = tmp[i] * _J;
  }
}
