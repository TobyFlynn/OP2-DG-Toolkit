inline void copy_dg_np_tk(const DG_FP *in, DG_FP *out) {
  for(int i = 0; i < DG_NP; i++) {
    out[i] = in[i];
  }
}
