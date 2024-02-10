inline void one_np(DG_FP *u) {
  for(int i = 0; i < DG_NP; i++) {
    u[i] = 1.0;
  }
}