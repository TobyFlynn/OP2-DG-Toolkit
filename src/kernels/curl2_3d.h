inline void curl2_3d(const double *dr, const double *ds, const double *dt,
                     const double *rx, const double *sx, const double *tx,
                     const double *ry, const double *sy, const double *ty,
                     double *resx, double *resy) {
  for(int i = 0; i < DG_NP; i++) {
    resx[i] += *ry * dr[i] + *sy * ds[i] + *ty * dt[i];
    resy[i] += -(*rx * dr[i] + *sx * ds[i] + *tx * dt[i]);
  }
}