inline void copy_dg_np_sp2dp_tk(const float *in, DG_FP *out) {
  for(int i = 0; i < DG_NP; i++) {
    out[i] = (DG_FP)in[i];
  }
}
