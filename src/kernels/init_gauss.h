inline void init_gauss(const int *p, DG_FP *rx, DG_FP *sx, DG_FP *ry,
                       DG_FP *sy, DG_FP *nx, DG_FP *ny, DG_FP *sJ) {
  // Get constants for this element's order
  const int dg_gnp  = DG_CONSTANTS_TK[(*p - 1) * 5 + 3];
  const int dg_gnfp = DG_CONSTANTS_TK[(*p - 1) * 5 + 4];
  // J = -xs.*yr + xr.*ys
  DG_FP J[DG_G_NP];
  for(int i = 0; i < dg_gnp; i++) {
    J[i] = -sx[i] * ry[i] + rx[i] * sy[i];
  }

  // rx = ys./J; sx =-yr./J; ry =-xs./J; sy = xr./J;
  for(int i = 0; i < dg_gnp; i++) {
    DG_FP rx_n = sy[i] / J[i];
    DG_FP sx_n = -ry[i] / J[i];
    DG_FP ry_n = -sx[i] / J[i];
    DG_FP sy_n = rx[i] / J[i];
    rx[i] = rx_n;
    sx[i] = sx_n;
    ry[i] = ry_n;
    sy[i] = sy_n;
  }

  // Calculate normals
  // Face 0
  for(int i = 0; i < dg_gnfp; i++) {
    nx[i] = -sx[i];
    ny[i] = -sy[i];
  }
  // Face 1
  for(int i = dg_gnfp; i < 2 * dg_gnfp; i++) {
    nx[i] = rx[i] + sx[i];
    ny[i] = ry[i] + sy[i];
  }
  // Face 2
  for(int i = 2 * dg_gnfp; i < dg_gnp; i++) {
    nx[i] = -rx[i];
    ny[i] = -ry[i];
  }

  // Normalise
  for(int i = 0; i < dg_gnp; i++) {
    sJ[i] = sqrt(nx[i] * nx[i] + ny[i] * ny[i]);
    nx[i] = nx[i] / sJ[i];
    ny[i] = ny[i] / sJ[i];
    sJ[i] = sJ[i] * J[i];
  }
}
