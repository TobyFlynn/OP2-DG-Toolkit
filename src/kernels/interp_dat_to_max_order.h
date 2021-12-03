inline void interp_dat_to_max_order(const double *mats, const int *order,
                                    const double *in, double *out) {
  // Get constants
  if(*order == DG_ORDER)
    return;

  const int dg_np_old = DG_CONSTANTS[(*order - 1) * 5];
  const int dg_np_new = DG_CONSTANTS[(DG_ORDER - 1) * 5];
  const double *mat = &mats[((*order - 1) * DG_ORDER + (DG_ORDER - 1)) * DG_NP * DG_NP];

  for(int i = 0; i < dg_np_new; i++) {
    out[i] = 0.0;
    for(int j = 0; j < dg_np_old; j++) {
      int ind = i + j * dg_np_new;
      out[i] += mat[ind] * in[j];
    }
  }
}
