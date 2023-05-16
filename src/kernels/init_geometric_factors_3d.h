inline void init_geometric_factors_3d(DG_FP *rx, DG_FP *ry, DG_FP *rz,
                                      DG_FP *sx, DG_FP *sy, DG_FP *sz,
                                      DG_FP *tx, DG_FP *ty, DG_FP *tz,
                                      DG_FP *J, DG_FP *geof) {
  DG_FP xr = *rx;
  DG_FP yr = *ry;
  DG_FP zr = *rz;
  DG_FP xs = *sx;
  DG_FP ys = *sy;
  DG_FP zs = *sz;
  DG_FP xt = *tx;
  DG_FP yt = *ty;
  DG_FP zt = *tz;
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
  geof[J_IND] = *J;
  geof[RX_IND] = *rx;
  geof[RY_IND] = *ry;
  geof[RZ_IND] = *rz;
  geof[SX_IND] = *sx;
  geof[SY_IND] = *sy;
  geof[SZ_IND] = *sz;
  geof[TX_IND] = *tx;
  geof[TY_IND] = *ty;
  geof[TZ_IND] = *tz;
}
