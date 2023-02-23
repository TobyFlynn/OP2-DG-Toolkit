inline void grad_2d(const int *p, const DG_FP *dr, const DG_FP *ds,
                    const DG_FP *u, const DG_FP *rx, const DG_FP *sx,
                    const DG_FP *ry, const DG_FP *sy, DG_FP *ux,
                    DG_FP *uy) {
  const DG_FP *dr_mat = &dr[(*p - 1) * DG_NP * DG_NP];
  const DG_FP *ds_mat = &ds[(*p - 1) * DG_NP * DG_NP];
  const int dg_np = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS];

  for(int m = 0; m < dg_np; m++) {
    DG_FP tmp_r = 0.0;
    DG_FP tmp_s = 0.0;
    for(int n = 0; n < dg_np; n++) {
      // int ind = m + n * dg_np;
      int ind = DG_MAT_IND(m, n, dg_np, dg_np);
      tmp_r += dr_mat[ind] * u[n];
      tmp_s += ds_mat[ind] * u[n];
    }
    ux[m] = rx[m] * tmp_r + sx[m] * tmp_s;
    uy[m] = ry[m] * tmp_r + sy[m] * tmp_s;
  }
}
