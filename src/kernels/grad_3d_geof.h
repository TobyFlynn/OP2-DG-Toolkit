inline void grad_3d_geof(const int *p, const DG_FP *geof, const DG_FP *ur,
                         const DG_FP *us, const DG_FP *ut, DG_FP *ux, DG_FP *uy,
                         DG_FP *uz) {
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  for(int m = 0; m < dg_np; m++) {
    ux[m] = geof[RX_IND] * ur[m] + geof[SX_IND] * us[m] + geof[TX_IND] * ut[m];
    uy[m] = geof[RY_IND] * ur[m] + geof[SY_IND] * us[m] + geof[TY_IND] * ut[m];
    uz[m] = geof[RZ_IND] * ur[m] + geof[SZ_IND] * us[m] + geof[TZ_IND] * ut[m];
  }
}
