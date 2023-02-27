inline void gemv_lift(const int *p, const DG_FP *alpha, const DG_FP *beta,
                      const DG_FP *matrix, const DG_FP *x, DG_FP *y) {
  // Get constants
  const int dg_np    = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS];
  const int dg_nfp   = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS + 1];
  const DG_FP *lift = &matrix[(*p - 1) * DG_NUM_FACES * DG_NPF * DG_NP];

  op2_in_kernel_gemv(false, dg_np, DG_NUM_FACES * dg_nfp, *alpha, lift, dg_np, x, *beta, y);
}
