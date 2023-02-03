inline void curl1_3d(const double *dr, const double *ds, const double *dt,
                     const double *rx, const double *sx, const double *tx,
                     const double *rz, const double *sz, const double *tz,
                     double *resx, double *resz) {
  for(int i = 0; i < DG_NP; i++) {
    resx[i] = -(*rz * dr[i] + *sz * ds[i] + *tz * dt[i]);
    resz[i] += *rx * dr[i] + *sx * ds[i] + *tx * dt[i];
  }
}