inline void curl0_3d(const DG_FP *dr, const DG_FP *ds, const DG_FP *dt,
                     const DG_FP *ry, const DG_FP *sy, const DG_FP *ty,
                     const DG_FP *rz, const DG_FP *sz, const DG_FP *tz,
                     DG_FP *resy, DG_FP *resz) {
  for(int i = 0; i < DG_NP; i++) {
    resy[i] = *rz * dr[i] + *sz * ds[i] + *tz * dt[i];
    resz[i] = -(*ry * dr[i] + *sy * ds[i] + *ty * dt[i]);
  }
}