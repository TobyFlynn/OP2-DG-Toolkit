inline void interp_dat_to_new_order(const double *mats, const int *old_order,
                                    const int *new_order, double *dat) {
  // Get constants
  if(*old_order == *new_order)
    return;

  const int dg_np_old = DG_CONSTANTS[(*old_order - 1) * 5];
  const int dg_np_new = DG_CONSTANTS[(*new_order - 1) * 5];
  const double *mat = &mats[((*old_order - 1) * DG_ORDER + (*new_order - 1)) * DG_NP * DG_NP];

  double res[DG_NP];

  for(int i = 0; i < dg_np_new; i++) {
    res[i] = 0.0;
    for(int j = 0; j < dg_np_old; j++) {
      int ind = i + j * dg_np_new;
      res[i] += mat[ind] * dat[j];
    }
  }

  for(int i = 0; i < dg_np_new; i++) {
    dat[i] = res[i];
  }
}
