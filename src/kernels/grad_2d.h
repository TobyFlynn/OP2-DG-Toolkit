inline void grad_2d(const int *p, const DG_FP *dr, const DG_FP *ds,
                    const DG_FP *u, const DG_FP *rx, const DG_FP *sx,
                    const DG_FP *ry, const DG_FP *sy, DG_FP *ux,
                    DG_FP *uy) {
  const DG_FP *dr_mat = &dr[(*p - 1) * DG_NP * DG_NP];
  const DG_FP *ds_mat = &ds[(*p - 1) * DG_NP * DG_NP];
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  DG_FP tmp_r[DG_NP], tmp_s[DG_NP];
  op2_in_kernel_gemv(false, dg_np, dg_np, 1.0, dr_mat, dg_np, u, 0.0, tmp_r);
  op2_in_kernel_gemv(false, dg_np, dg_np, 1.0, ds_mat, dg_np, u, 0.0, tmp_s);

  for(int m = 0; m < dg_np; m++) {
    ux[m] = rx[m] * tmp_r[m] + sx[m] * tmp_s[m];
    uy[m] = ry[m] * tmp_r[m] + sy[m] * tmp_s[m];
  }
}
