inline void copy_dg_np_sp_tk(const float *in, float *out) {
  for(int i = 0; i < DG_NP; i++) {
    out[i] = in[i];
  }
}
