inline void curl0_3d(const DG_FP *dr, const DG_FP *ds, const DG_FP *dt,
                     const DG_FP *geof, DG_FP *resy, DG_FP *resz) {
  for(int i = 0; i < DG_NP; i++) {
    resy[i] = geof[RZ_IND] * dr[i] + geof[SZ_IND] * ds[i] + geof[TZ_IND] * dt[i];
    resz[i] = -(geof[RY_IND] * dr[i] + geof[SY_IND] * ds[i] + geof[TY_IND] * dt[i]);
  }
}