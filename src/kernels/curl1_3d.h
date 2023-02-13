inline void curl1_3d(const DG_FP *dr, const DG_FP *ds, const DG_FP *dt,
                     const DG_FP *rx, const DG_FP *sx, const DG_FP *tx,
                     const DG_FP *rz, const DG_FP *sz, const DG_FP *tz,
                     DG_FP *resx, DG_FP *resz) {
  for(int i = 0; i < DG_NP; i++) {
    resx[i] = -(*rz * dr[i] + *sz * ds[i] + *tz * dt[i]);
    resz[i] += *rx * dr[i] + *sx * ds[i] + *tx * dt[i];
  }
}