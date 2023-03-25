inline void div_3d(const int *p, const DG_FP *dr, const DG_FP *ds,
                   const DG_FP *dt, const DG_FP *u, const DG_FP *v, const DG_FP *w,
                   const DG_FP *rx, const DG_FP *sx, const DG_FP *tx,
                   const DG_FP *ry, const DG_FP *sy, const DG_FP *ty,
                   const DG_FP *rz, const DG_FP *sz, const DG_FP *tz,
                   DG_FP *out) {
  const DG_FP *dr_mat = &dr[(*p - 1) * DG_NP * DG_NP];
  const DG_FP *ds_mat = &ds[(*p - 1) * DG_NP * DG_NP];
  const DG_FP *dt_mat = &dt[(*p - 1) * DG_NP * DG_NP];
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  DG_FP tmp[DG_NP];
  for(int n = 0; n < dg_np; n++) {
    tmp[n] = *rx * u[n] + *ry * v[n] + *rz * w[n];
  }

  op2_in_kernel_gemv(false, dg_np, dg_np, 1.0, dr_mat, dg_np, tmp, 0.0, out);

  for(int n = 0; n < dg_np; n++) {
    tmp[n] = *sx * u[n] + *sy * v[n] + *sz * w[n];
  }

  op2_in_kernel_gemv(false, dg_np, dg_np, 1.0, ds_mat, dg_np, tmp, 1.0, out);

  for(int n = 0; n < dg_np; n++) {
    tmp[n] = *tx * u[n] + *ty * v[n] + *tz * w[n];
  }

  op2_in_kernel_gemv(false, dg_np, dg_np, 1.0, dt_mat, dg_np, tmp, 1.0, out);
}
