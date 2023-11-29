inline void copy_dg_np_dp2sp_tk(const DG_FP * __restrict__ in, float * __restrict__ out) {
  for(int i = 0; i < DG_NP; i++) {
    out[i] = (float)in[i];
  }
}
