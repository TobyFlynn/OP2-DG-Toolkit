inline void pmf_2d_mult_mm(const int *p, const DG_FP *mass,
                           const DG_FP *J, const DG_FP *mm_factor,
                           const DG_FP *in, DG_FP *out) {
  const DG_FP *mass_mat = &mass[(*p - 1) * DG_NP * DG_NP];
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  // TODO check if this should be transpose
  for(int m = 0; m < dg_np; m++) {
    for(int n = 0; n < dg_np; n++) {
      // int ind = m * dg_np + n;
      int ind = DG_MAT_IND(m, n, dg_np, dg_np);
      out[m] += *mm_factor * J[0] * mass_mat[ind] * in[n];
    }
  }
}
