inline void p_multigrid_relaxation_chebyshev_2(const int *order, const float *factor0, const float *factor1,
                                               const float *b, const float *diag, float *res, float *d,
                                               float *u) {
  const int dg_np = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS];
  const float _factor0 = *factor0;
  const float _factor1 = *factor1;
  for(int i = 0; i < dg_np; i++) {
    res[i] -= b[i] / diag[i];
    d[i] = _factor0 * res[i] + _factor1 * d[i];
    u[i] += d[i];
  }
}
