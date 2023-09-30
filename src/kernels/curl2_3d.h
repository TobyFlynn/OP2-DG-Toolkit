inline void curl2_3d(const DG_FP *dr, const DG_FP *ds, const DG_FP *dt,
                     const DG_FP *geof, DG_FP *resx, DG_FP *resy) {
  for(int i = 0; i < DG_NP; i++) {
    resx[i] += geof[RY_IND] * dr[i] + geof[SY_IND] * ds[i] + geof[TY_IND] * dt[i];
    resy[i] += -(geof[RX_IND] * dr[i] + geof[SX_IND] * ds[i] + geof[TX_IND] * dt[i]);
  }
}