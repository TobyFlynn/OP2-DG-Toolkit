inline void J(const int *p, const DG_FP *geof, const DG_FP *tmp, DG_FP *u) {
  // Get constants for this element's order
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];
  const DG_FP J = geof[J_IND];
  for(int i = 0; i < dg_np; i++) {
    u[i] = tmp[i] * J;
  }
}
