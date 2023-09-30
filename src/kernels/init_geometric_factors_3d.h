inline void init_geometric_factors_3d(const DG_FP *x, const DG_FP *y, const DG_FP *z,
                                      DG_FP *geof) {
  const DG_FP *dr_mat = &dg_Dr_kernel[(DG_ORDER - 1) * DG_NP * DG_NP];
  const DG_FP *ds_mat = &dg_Ds_kernel[(DG_ORDER - 1) * DG_NP * DG_NP];
  const DG_FP *dt_mat = &dg_Dt_kernel[(DG_ORDER - 1) * DG_NP * DG_NP];

  DG_FP xr = 0.0;
  DG_FP yr = 0.0;
  DG_FP zr = 0.0;
  for(int i = 0; i < DG_NP; i++) {
    int ind = DG_MAT_IND(0, i, DG_NP, DG_NP);
    xr += dr_mat[ind] * x[i];
    yr += dr_mat[ind] * y[i];
    zr += dr_mat[ind] * z[i];
  }

  DG_FP xs = 0.0;
  DG_FP ys = 0.0;
  DG_FP zs = 0.0;
  for(int i = 0; i < DG_NP; i++) {
    int ind = DG_MAT_IND(0, i, DG_NP, DG_NP);
    xs += ds_mat[ind] * x[i];
    ys += ds_mat[ind] * y[i];
    zs += ds_mat[ind] * z[i];
  }

  DG_FP xt = 0.0;
  DG_FP yt = 0.0;
  DG_FP zt = 0.0;
  for(int i = 0; i < DG_NP; i++) {
    int ind = DG_MAT_IND(0, i, DG_NP, DG_NP);
    xt += dt_mat[ind] * x[i];
    yt += dt_mat[ind] * y[i];
    zt += dt_mat[ind] * z[i];
  }

  const DG_FP J  = xr * (ys * zt - zs * yt) - yr * (xs * zt - zs * xt) + zr * (xs * yt - ys * xt);
  const DG_FP rx = (ys * zt - zs * yt) / J;
  const DG_FP ry = -(xs * zt - zs * xt) / J;
  const DG_FP rz = (xs * yt - ys * xt) / J;
  const DG_FP sx = -(yr * zt - zr * yt) / J;
  const DG_FP sy = (xr * zt - zr * xt) / J;
  const DG_FP sz = -(xr * yt - yr * xt) / J;
  const DG_FP tx = (yr * zs - zr * ys) / J;
  const DG_FP ty = -(xr * zs - zr * xs) / J;
  const DG_FP tz = (xr * ys - yr * xs) / J;

  geof[J_IND] = J;
  geof[RX_IND] = rx;
  geof[RY_IND] = ry;
  geof[RZ_IND] = rz;
  geof[SX_IND] = sx;
  geof[SY_IND] = sy;
  geof[SZ_IND] = sz;
  geof[TX_IND] = tx;
  geof[TY_IND] = ty;
  geof[TZ_IND] = tz;
}
