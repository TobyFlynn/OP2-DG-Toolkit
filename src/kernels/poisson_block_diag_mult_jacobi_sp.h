inline void poisson_block_diag_mult_jacobi_sp(const int *p, const DG_FP *diag, float *rhs) {
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];
  for(int i = 0; i < dg_np; i++) {
    const float diag_tmp = (float)diag[i * dg_np + i];
    rhs[i] = rhs[i] / diag_tmp;
  }
}