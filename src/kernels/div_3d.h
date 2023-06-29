inline void div_3d(const int *p, const DG_FP *u, const DG_FP *v, const DG_FP *w,
                   const DG_FP *geof, DG_FP *out0, DG_FP *out1, DG_FP *out2) {
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  for(int n = 0; n < dg_np; n++) {
    out0[n] = geof[RX_IND] * u[n] + geof[RY_IND] * v[n] + geof[RZ_IND] * w[n];
    out1[n] = geof[SX_IND] * u[n] + geof[SY_IND] * v[n] + geof[SZ_IND] * w[n];
    out2[n] = geof[TX_IND] * u[n] + geof[TY_IND] * v[n] + geof[TZ_IND] * w[n];
  }
}
