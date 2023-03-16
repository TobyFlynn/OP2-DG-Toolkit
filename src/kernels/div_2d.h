inline void div_2d(const int *p, const DG_FP *dr, const DG_FP *ds,
                   const DG_FP *u, const DG_FP *v, const DG_FP *rx,
                   const DG_FP *sx, const DG_FP *ry, const DG_FP *sy,
                   DG_FP *out) {
  const DG_FP *dr_mat = &dr[(*p - 1) * DG_NP * DG_NP];
  const DG_FP *ds_mat = &ds[(*p - 1) * DG_NP * DG_NP];
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  DG_FP tmp_r[DG_NP], tmp_s[DG_NP];
  for(int n = 0; n < dg_np; n++) {
    tmp_r[n] = rx[n] * u[n] + ry[n] * v[n];
    tmp_s[n] = sx[n] * u[n] + sy[n] * v[n];
  }

  op2_in_kernel_gemv(false, dg_np, dg_np, 1.0, dr_mat, dg_np, tmp_r, 0.0, out);
  op2_in_kernel_gemv(false, dg_np, dg_np, 1.0, ds_mat, dg_np, tmp_s, 1.0, out);
}
