inline void curl1_3d(const DG_FP *dr, const DG_FP *ds, const DG_FP *dt,
                     const DG_FP *geof, DG_FP *resx, DG_FP *resz) {
  for(int i = 0; i < DG_NP; i++) {
    resx[i] = -(geof[RZ_IND] * dr[i] + geof[SZ_IND] * ds[i] + geof[TZ_IND] * dt[i]);
    resz[i] += geof[RX_IND] * dr[i] + geof[SX_IND] * ds[i] + geof[TX_IND] * dt[i];
  }
}