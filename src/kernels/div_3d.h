inline void div_3d(const int *p, const DG_FP *u, const DG_FP *v, const DG_FP *w,
                   const DG_FP *rx, const DG_FP *sx, const DG_FP *tx,
                   const DG_FP *ry, const DG_FP *sy, const DG_FP *ty,
                   const DG_FP *rz, const DG_FP *sz, const DG_FP *tz,
                   DG_FP *out0, DG_FP *out1, DG_FP *out2) {
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  for(int n = 0; n < dg_np; n++) {
    out0[n] = *rx * u[n] + *ry * v[n] + *rz * w[n];
    out1[n] = *sx * u[n] + *sy * v[n] + *sz * w[n];
    out2[n] = *tx * u[n] + *ty * v[n] + *tz * w[n];
  }
}
