inline void init_grid(const int *p, DG_FP *rx, DG_FP *ry, DG_FP *sx,
                      DG_FP *sy, DG_FP *nx, DG_FP *ny, DG_FP *J, DG_FP *sJ,
                      DG_FP *fscale, DG_FP *nx_c, DG_FP *ny_c, DG_FP *sJ_c,
                      DG_FP *fscale_c) {
  // Get constants for this element's order
  const int *fmask = &FMASK_TK[(*p - 1) * 3 * DG_NPF];
  const int dg_np  = DG_CONSTANTS_TK[(*p - 1) * 5];
  const int dg_nfp = DG_CONSTANTS_TK[(*p - 1) * 5 + 1];
  // Calculate normals
  // Face 0
  for(int i = 0; i < dg_nfp; i++) {
    const int fmask_ind = fmask[i];
    nx[i] = ry[fmask_ind];
    ny[i] = -rx[fmask_ind];
  }
  // Face 1
  for(int i = 0; i < dg_nfp; i++) {
    const int fmask_ind = fmask[dg_nfp + i];
    nx[dg_nfp + i] = sy[fmask_ind] - ry[fmask_ind];
    ny[dg_nfp + i] = rx[fmask_ind] - sx[fmask_ind];
  }
  // Face 2
  for(int i = 0; i < dg_nfp; i++) {
    const int fmask_ind = fmask[2 * dg_nfp + i];
    nx[2 * dg_nfp + i] = -sy[fmask_ind];
    ny[2 * dg_nfp + i] = sx[fmask_ind];
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
    const int fmask_ind = fmask[i];
    // sJ[i] /= 2.0;
    fscale[i] = sJ[i] / J[fmask_ind];
  }

  for(int i = 0; i < 3; i++) {
    nx_c[i] = nx[i * dg_nfp];
    ny_c[i] = ny[i * dg_nfp];
    sJ_c[i] = sJ[i * dg_nfp];
    fscale_c[i] = fscale[i * dg_nfp];
  }
}
