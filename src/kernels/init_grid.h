inline void init_grid(double *rx, double *ry, double *sx, double *sy,
                      double *nx, double *ny, double *J, double *sJ,
                      double *fscale) {
  // Calculate normals
  // Face 0
  for(int i = 0; i < DG_NPF; i++) {
    nx[i] = ry[FMASK[i]];
    ny[i] = -rx[FMASK[i]];
  }
  // Face 1
  for(int i = 0; i < DG_NPF; i++) {
    nx[DG_NPF + i] = sy[FMASK[DG_NPF + i]] - ry[FMASK[DG_NPF + i]];
    ny[DG_NPF + i] = rx[FMASK[DG_NPF + i]] - sx[FMASK[DG_NPF + i]];
  }
  // Face 2
  for(int i = 0; i < DG_NPF; i++) {
    nx[2 * DG_NPF + i] = -sy[FMASK[2 * DG_NPF + i]];
    ny[2 * DG_NPF + i] = sx[FMASK[2 * DG_NPF + i]];
  }

  // J = -xs.*yr + xr.*ys
  for(int i = 0; i < DG_NP; i++) {
    J[i] = -sx[i] * ry[i] + rx[i] * sy[i];
  }

  // rx = ys./J; sx =-yr./J; ry =-xs./J; sy = xr./J;
  for(int i = 0; i < DG_NP; i++) {
    double rx_n = sy[i] / J[i];
    double sx_n = -ry[i] / J[i];
    double ry_n = -sx[i] / J[i];
    double sy_n = rx[i] / J[i];
    rx[i] = rx_n;
    sx[i] = sx_n;
    ry[i] = ry_n;
    sy[i] = sy_n;
  }

  // Normalise
  for(int i = 0; i < 3 * DG_NPF; i++) {
    sJ[i] = sqrt(nx[i] * nx[i] + ny[i] * ny[i]);
    nx[i] = nx[i] / sJ[i];
    ny[i] = ny[i] / sJ[i];
    fscale[i] = sJ[i] / J[FMASK[i]];
  }
}
