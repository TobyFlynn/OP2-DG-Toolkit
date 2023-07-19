inline void p_multigrid_relaxation_chebyshev_1(const int *order, const float *in,
                                               float *out) {
  const int dg_np = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS];
  for(int i = 0; i < dg_np; i++) {
    out[i] += in[i];
  }
}
