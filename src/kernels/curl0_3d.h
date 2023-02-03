inline void curl0_3d(const double *dr, const double *ds, const double *dt,
                     const double *ry, const double *sy, const double *ty,
                     const double *rz, const double *sz, const double *tz,
                     double *resy, double *resz) {
  for(int i = 0; i < DG_NP; i++) {
    resy[i] = *rz * dr[i] + *sz * ds[i] + *tz * dt[i];
    resz[i] = -(*ry * dr[i] + *sy * ds[i] + *ty * dt[i]);
  }
}