inline void p_multigrid_restriction(const int *p, const float *Au,
                                    const float *f, float *b, float *u) {
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];
  for(int i = 0; i < dg_np; i++) {
    b[i] = f[i] - Au[i];
    u[i] = 0.0f;
  }
}
