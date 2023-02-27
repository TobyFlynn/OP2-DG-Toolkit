inline void gemv_cub_np_npT(const int *p, const DG_FP *alpha,
                            const DG_FP *beta, const DG_FP *matrix,
                            const DG_FP *x, DG_FP *y) {
  // Get constants
  const int dg_np     = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS];
  const int dg_cub_np = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS + 2];
  const DG_FP *mat    = &matrix[(*p - 1) * DG_CUB_NP * DG_NP];

  op2_in_kernel_gemv(true, dg_cub_np, dg_np, *alpha, mat, dg_cub_np, x, *beta, y);
}
