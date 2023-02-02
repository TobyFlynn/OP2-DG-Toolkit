inline void init_geometric_factors_copy_3d(const double *xr, const double *xs,
                                           const double *xt, double *rx,
                                           double *sx, double *tx) {
  *rx = xr[0];
  *sx = xs[0];
  *tx = xt[0];
}