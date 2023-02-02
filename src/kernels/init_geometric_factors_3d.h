inline void init_geometric_factors_3d(double *rx, double *ry, double *rz,
                                      double *sx, double *sy, double *sz,
                                      double *tx, double *ty, double *tz,
                                      double *J) {
  double xr = *rx;
  double yr = *ry;
  double zr = *rz;
  double xs = *sx;
  double ys = *sy;
  double zs = *sz;
  double xt = *tx;
  double yt = *ty;
  double zt = *tz;
  *J  = xr * (ys * zt - zs * yt) - yr * (xs * zt - zs * xt) + zr * (xs * yt - ys * xt);
  *rx = (ys * zt - zs * yt) / *J;
  *ry = -(xs * zt - zs * xt) / *J;
  *rz = (xs * yt - ys * xt) / *J;
  *sx = -(yr * zt - zr * yt) / *J;
  *sy = (xr * zt - zr * xt) / *J;
  *sz = -(xr * yt - yr * xt) / *J;
  *tx = (yr * zs - zr * ys) / *J;
  *ty = -(xr * zs - zr * xs) / *J;
  *tz = (xr * ys - yr * xs) / *J;
}