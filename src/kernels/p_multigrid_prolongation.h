inline void p_multigrid_prolongation(const int *p, const float *u_old, float *u) {
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];
  for(int i = 0; i < dg_np; i++) {
    u[i] += u_old[i];
  }
}
