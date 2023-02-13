inline void init_geometric_factors_copy_3d(const DG_FP *xr, const DG_FP *xs,
                                           const DG_FP *xt, DG_FP *rx,
                                           DG_FP *sx, DG_FP *tx) {
  *rx = xr[0];
  *sx = xs[0];
  *tx = xt[0];
}