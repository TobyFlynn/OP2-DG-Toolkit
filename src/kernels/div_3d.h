inline void div_3d(const double *dr, const double *ds, const double *dt,
                const double *rc, const double *sc, const double *tc,
                double *res) {
  for(int i = 0; i < DG_NP; i++) {
    res[i] += *rc * dr[i] + *sc * ds[i] + *tc * dt[i];
  }
}