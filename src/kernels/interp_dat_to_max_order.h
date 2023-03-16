inline void interp_dat_to_max_order(const DG_FP *mats, const int *order,
                                    const DG_FP *in, DG_FP *out) {
  // Get constants
  const int dg_np_old = DG_CONSTANTS_TK[(*order - 1) * 5];
  const int dg_np_new = DG_CONSTANTS_TK[(DG_ORDER - 1) * 5];
  const DG_FP *mat = &mats[((*order - 1) * DG_ORDER + (DG_ORDER - 1)) * DG_NP * DG_NP];

  if(*order == DG_ORDER) {
    for(int i = 0; i < dg_np_new; i++) {
      out[i] = in[i];
    }
    return;
  }

  op2_in_kernel_gemv(false, dg_np_new, dg_np_old, 1.0, mat, dg_np_new, in, 0.0, out);
/*
  for(int i = 0; i < dg_np_new; i++) {
    out[i] = 0.0;
    for(int j = 0; j < dg_np_old; j++) {
      // int ind = i + j * dg_np_new;
      int ind = DG_MAT_IND(i, j, dg_np_new, dg_np_old);
      out[i] += mat[ind] * in[j];
    }
  }
*/
}
