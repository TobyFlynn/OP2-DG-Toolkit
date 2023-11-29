inline void fpmf_3d_mult_mm(const int * __restrict__ p, const DG_FP * __restrict__ geof, const DG_FP * __restrict__ mm_factor,
                            const DG_FP * __restrict__ in, DG_FP * __restrict__ out) {
  const int _p = *p;
  const DG_FP *mass_mat = &dg_Mass_kernel[(_p - 1) * DG_NP * DG_NP];
  const int dg_np = DG_CONSTANTS_TK[(_p - 1) * DG_NUM_CONSTANTS];

  // TODO don't think this should be transpose
  const DG_FP _J = geof[J_IND];
  for(int m = 0; m < dg_np; m++) {
    DG_FP tmp = 0.0;
    for(int n = 0; n < dg_np; n++) {
      // int ind = m * dg_np + n;
      int ind = DG_MAT_IND(m, n, dg_np, dg_np);
      tmp += mm_factor[n] * mass_mat[ind] * in[n];
    }
    out[m] += tmp * _J;
  }
}
