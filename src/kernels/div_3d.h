inline void div_3d(const int *p, const DG_FP *dr, const DG_FP *ds,
                   const DG_FP *dt, const DG_FP *u, const DG_FP *v, const DG_FP *w,
                   const DG_FP *rx, const DG_FP *sx, const DG_FP *tx,
                   const DG_FP *ry, const DG_FP *sy, const DG_FP *ty,
                   const DG_FP *rz, const DG_FP *sz, const DG_FP *tz,
                   DG_FP *out) {
  const DG_FP *dr_mat = &dr[(*p - 1) * DG_NP * DG_NP];
  const DG_FP *ds_mat = &ds[(*p - 1) * DG_NP * DG_NP];
  const DG_FP *dt_mat = &dt[(*p - 1) * DG_NP * DG_NP];
  const int dg_np = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS];

  for(int m = 0; m < dg_np; m++) {
    out[m] = 0.0;
  }

  for(int m = 0; m < dg_np; m++) {
    for(int n = 0; n < dg_np; n++) {
      // int ind = m + n * dg_np;
      int ind = DG_MAT_IND(m, n, dg_np, dg_np);
      out[m] += dr_mat[ind] * (*rx * u[n] + *ry * v[n] + *rz * w[n]);
      out[m] += ds_mat[ind] * (*sx * u[n] + *sy * v[n] + *sz * w[n]);
      out[m] += dt_mat[ind] * (*tx * u[n] + *ty * v[n] + *tz * w[n]);
    }
  }
}
