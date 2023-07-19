inline void poisson_diag_mult_jacobi_sp(const int *p, const DG_FP *diag, float *rhs) {
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];
  for(int i = 0; i < dg_np; i++) {
    rhs[i] = rhs[i] / (float)diag[i];
  }
}
