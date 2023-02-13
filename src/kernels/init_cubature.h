inline void init_cubature(const int *p, const DG_FP *matrix, DG_FP *rx,
                          DG_FP *sx, DG_FP *ry, DG_FP *sy, DG_FP *J,
                          DG_FP *temp) {
  // Get constants for this element's order
  const int dg_np      = DG_CONSTANTS[(*p - 1) * 5];
  const int dg_cub_np  = DG_CONSTANTS[(*p - 1) * 5 + 2];
  const DG_FP *cubW = &cubW_g[(*p - 1) * DG_CUB_NP];
  const DG_FP *cubV = &matrix[(*p - 1) * DG_CUB_NP * DG_NP];

  // J = -xs.*yr + xr.*ys
  for(int i = 0; i < dg_cub_np; i++) {
    J[i] = -sx[i] * ry[i] + rx[i] * sy[i];
  }

  // rx = ys./J; sx =-yr./J; ry =-xs./J; sy = xr./J;
  for(int i = 0; i < dg_cub_np; i++) {
    DG_FP rx_n = sy[i] / J[i];
    DG_FP sx_n = -ry[i] / J[i];
    DG_FP ry_n = -sx[i] / J[i];
    DG_FP sy_n = rx[i] / J[i];
    rx[i] = rx_n;
    sx[i] = sx_n;
    ry[i] = ry_n;
    sy[i] = sy_n;
  }

  for(int j = 0; j < dg_np; j++) {
    for(int i = 0; i < dg_cub_np; i++) {
      int ind = j * dg_cub_np + i;
      temp[ind] = J[i] * cubW[i] * cubV[ind];
    }
  }
}
