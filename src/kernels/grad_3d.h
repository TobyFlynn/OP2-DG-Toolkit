inline void grad_3d(const int *p, const DG_FP *dr, const DG_FP *ds,
                    const DG_FP *dt, const DG_FP *u,
                    const DG_FP *rx, const DG_FP *sx, const DG_FP *tx,
                    const DG_FP *ry, const DG_FP *sy, const DG_FP *ty,
                    const DG_FP *rz, const DG_FP *sz, const DG_FP *tz,
                    DG_FP *ux, DG_FP *uy, DG_FP *uz) {
  const DG_FP *dr_mat = &dr[(*p - 1) * DG_NP * DG_NP];
  const DG_FP *ds_mat = &ds[(*p - 1) * DG_NP * DG_NP];
  const DG_FP *dt_mat = &dt[(*p - 1) * DG_NP * DG_NP];
  const int dg_np = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS];

  for(int m = 0; m < dg_np; m++) {
    DG_FP tmp_r = 0.0;
    DG_FP tmp_s = 0.0;
    DG_FP tmp_t = 0.0;
    for(int n = 0; n < dg_np; n++) {
      // int ind = m + n * dg_np;
      int ind = DG_MAT_IND(m, n, dg_np, dg_np);
      tmp_r += dr_mat[ind] * u[n];
      tmp_s += ds_mat[ind] * u[n];
      tmp_t += dt_mat[ind] * u[n];
    }
    ux[m] = *rx * tmp_r + *sx * tmp_s + *tx * tmp_t;
    uy[m] = *ry * tmp_r + *sy * tmp_s + *ty * tmp_t;
    uz[m] = *rz * tmp_r + *sz * tmp_s + *tz * tmp_t;
  }
}
