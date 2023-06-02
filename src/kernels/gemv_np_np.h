inline void gemv_np_np(const int *p, const DG_FP *alpha, const DG_FP *beta,
                       const DG_FP *matrix, const DG_FP *x, DG_FP *y) {
  // Get constants
  const int dg_np  = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];
  const DG_FP *mat = &matrix[(*p - 1) * DG_NP * DG_NP];

  op2_in_kernel_gemv(false, dg_np, dg_np, *alpha, mat, dg_np, x, *beta, y);
}
