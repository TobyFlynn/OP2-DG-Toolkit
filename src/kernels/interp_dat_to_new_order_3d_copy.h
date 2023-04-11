inline void interp_dat_to_new_order_3d_copy(const DG_FP *mats, const int *old_order,
                                            const int *new_order, const DG_FP *in,
                                            DG_FP *out) {
  // Get constants
  const int dg_np_old = DG_CONSTANTS_TK[(*old_order - 1) * DG_NUM_CONSTANTS];
  const int dg_np_new = DG_CONSTANTS_TK[(*new_order - 1) * DG_NUM_CONSTANTS];

  if(*old_order == *new_order) {
    for(int i = 0; i < dg_np_old; i++) {
      out[i] = in[i];
    }
    return;
  }

  const DG_FP *mat = &mats[((*old_order - 1) * DG_ORDER + (*new_order - 1)) * DG_NP * DG_NP];

  op2_in_kernel_gemv(false, dg_np_new, dg_np_old, 1.0, mat, dg_np_new, in, 0.0, out);
}
