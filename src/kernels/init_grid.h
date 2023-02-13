inline void init_grid(const int *p, DG_FP *rx, DG_FP *ry, DG_FP *sx,
                      DG_FP *sy, DG_FP *nx, DG_FP *ny, DG_FP *J, DG_FP *sJ,
                      DG_FP *fscale) {
  // Get constants for this element's order
  const int *fmask = &FMASK[(*p - 1) * 3 * DG_NPF];
  const int dg_np  = DG_CONSTANTS[(*p - 1) * 5];
  const int dg_nfp = DG_CONSTANTS[(*p - 1) * 5 + 1];
  // Calculate normals
  // Face 0
  for(int i = 0; i < dg_nfp; i++) {
    nx[i] = ry[fmask[i]];
    ny[i] = -rx[fmask[i]];
  }
  // Face 1
  for(int i = 0; i < dg_nfp; i++) {
    nx[dg_nfp + i] = sy[fmask[dg_nfp + i]] - ry[fmask[dg_nfp + i]];
    ny[dg_nfp + i] = rx[fmask[dg_nfp + i]] - sx[fmask[dg_nfp + i]];
  }
  // Face 2
  for(int i = 0; i < dg_nfp; i++) {
    nx[2 * dg_nfp + i] = -sy[fmask[2 * dg_nfp + i]];
    ny[2 * dg_nfp + i] = sx[fmask[2 * dg_nfp + i]];
  }

  // J = -xs.*yr + xr.*ys
  for(int i = 0; i < dg_np; i++) {
    J[i] = -sx[i] * ry[i] + rx[i] * sy[i];
  }

  // rx = ys./J; sx =-yr./J; ry =-xs./J; sy = xr./J;
  for(int i = 0; i < dg_np; i++) {
    DG_FP rx_n = sy[i] / J[i];
    DG_FP sx_n = -ry[i] / J[i];
    DG_FP ry_n = -sx[i] / J[i];
    DG_FP sy_n = rx[i] / J[i];
    rx[i] = rx_n;
    sx[i] = sx_n;
    ry[i] = ry_n;
    sy[i] = sy_n;
  }

  // Normalise
  for(int i = 0; i < 3 * dg_nfp; i++) {
    sJ[i] = sqrt(nx[i] * nx[i] + ny[i] * ny[i]);
    nx[i] = nx[i] / sJ[i];
    ny[i] = ny[i] / sJ[i];
    fscale[i] = sJ[i] / J[fmask[i]];
  }
}
