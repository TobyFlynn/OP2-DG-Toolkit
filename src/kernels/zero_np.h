inline void zero_np(double *u) {
  for(int i = 0; i < DG_NP; i++) {
    u[i] = 0.0;
  }
}