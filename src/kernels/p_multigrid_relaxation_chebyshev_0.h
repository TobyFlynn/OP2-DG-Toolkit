inline void p_multigrid_relaxation_chebyshev_0(const float *factor,
                                const int *order, const float *b,
                                const float *diag, float *res, float *out,
                                float *u) {
  const int dg_np = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS];
  const float _factor = *factor;
  for(int i = 0; i < dg_np; i++) {
    res[i] = b[i] - res[i];
    res[i] /= diag[i];
    out[i] = _factor * res[i];
    u[i] += out[i];
  }
}
