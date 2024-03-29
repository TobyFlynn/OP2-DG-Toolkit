inline void p_multigrid_vec_normalise(const int *order, const float *in,
                                      const float *norm, float *out) {
  const int dg_np = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS];
  for(int i = 0; i < dg_np; i++) {
    out[i] = in[i] / *norm;
  }
}
