inline void inv_mass(const int *p, const DG_FP *J, DG_FP *x) {
  // Get constants
  const int dg_np  = DG_CONSTANTS_TK[(*p - 1) * 5];
  const DG_FP *mat = &dg_InvMass_kernel[(*p - 1) * DG_NP * DG_NP];

  DG_FP tmp[DG_NP];

  op2_in_kernel_gemv(false, dg_np, dg_np, 1.0, mat, dg_np, x, 0.0, tmp);

  for(int i = 0; i < dg_np; i++) {
    x[i] = tmp[i] / *J;
  }
}
