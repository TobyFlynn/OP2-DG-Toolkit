inline void p_multigrid_relaxation_chebyshev_3(const DG_FP *factor0, const DG_FP *factor1,
                                               const int *order, const DG_FP *a,
                                               DG_FP *b) {
  const int dg_np = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS];
  for(int i = 0; i < dg_np; i++) {
    b[i] = *factor0 * a[i] + *factor1 * b[i];
  }
}
