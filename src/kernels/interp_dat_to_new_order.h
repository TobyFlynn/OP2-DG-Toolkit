inline void interp_dat_to_new_order(const DG_FP *mats, const int *old_order,
                                    const int *new_order, DG_FP *dat) {
  // Get constants
  if(*old_order == *new_order)
    return;

  const int dg_np_old = DG_CONSTANTS_TK[(*old_order - 1) * 5];
  const int dg_np_new = DG_CONSTANTS_TK[(*new_order - 1) * 5];
  const DG_FP *mat = &mats[((*old_order - 1) * DG_ORDER + (*new_order - 1)) * DG_NP * DG_NP];

  DG_FP res[DG_NP];

  op2_in_kernel_gemv(false, dg_np_new, dg_np_old, 1.0, mat, dg_np_new, dat, 0.0, res);

/*
  for(int i = 0; i < dg_np_new; i++) {
    res[i] = 0.0;
    for(int j = 0; j < dg_np_old; j++) {
      // int ind = i + j * dg_np_new;
      int ind = DG_MAT_IND(i, j, dg_np_new, dg_np_old);
      res[i] += mat[ind] * dat[j];
    }
  }
*/

  for(int i = 0; i < dg_np_new; i++) {
    dat[i] = res[i];
  }
}
