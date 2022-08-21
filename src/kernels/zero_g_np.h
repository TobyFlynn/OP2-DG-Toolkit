inline void zero_g_np(double *g_np) {
  for(int i = 0; i < DG_G_NP; i++) {
    g_np[i] = 0.0;
  }
}