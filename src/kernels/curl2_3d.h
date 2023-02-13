inline void curl2_3d(const DG_FP *dr, const DG_FP *ds, const DG_FP *dt,
                     const DG_FP *rx, const DG_FP *sx, const DG_FP *tx,
                     const DG_FP *ry, const DG_FP *sy, const DG_FP *ty,
                     DG_FP *resx, DG_FP *resy) {
  for(int i = 0; i < DG_NP; i++) {
    resx[i] += *ry * dr[i] + *sy * ds[i] + *ty * dt[i];
    resy[i] += -(*rx * dr[i] + *sx * ds[i] + *tx * dt[i]);
  }
}