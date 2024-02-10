inline void one_np_sp(float *u) {
  for(int i = 0; i < DG_NP; i++) {
    u[i] = 1.0f;
  }
}