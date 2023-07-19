inline void p_multigrid_relaxation_chebyshev_3(const float *factor0, const float *factor1,
                                               const int *order, const float *a,
                                               float *b) {
  const int dg_np = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS];
  for(int i = 0; i < dg_np; i++) {
    b[i] = *factor0 * a[i] + *factor1 * b[i];
  }
}
