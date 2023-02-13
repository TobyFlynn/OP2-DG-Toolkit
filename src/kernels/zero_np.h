inline void zero_np(DG_FP *u) {
  for(int i = 0; i < DG_NP; i++) {
    u[i] = 0.0;
  }
}