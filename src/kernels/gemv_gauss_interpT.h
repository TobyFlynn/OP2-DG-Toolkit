inline void gemv_gauss_interpT(const int *p, const DG_FP *alpha,
                               const DG_FP *beta, const DG_FP *matrix,
                               const DG_FP *x, DG_FP *y) {
  // Get constants
  const int dg_np   = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];
  const int dg_g_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS + 3];
  const DG_FP *gauss_interp = &matrix[(*p - 1) * DG_G_NP * DG_NP];

  op2_in_kernel_gemv(true, dg_g_np, dg_np, *alpha, gauss_interp, dg_g_np, x, *beta, y);
}
