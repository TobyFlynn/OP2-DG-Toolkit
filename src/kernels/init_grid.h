inline void init_grid(const DG_FP *x, const DG_FP *y, DG_FP *nx, DG_FP *ny, 
                      DG_FP *sJ, DG_FP *fscale, DG_FP *geof) {
  const DG_FP *dr_mat = &dg_Dr_kernel[(DG_ORDER - 1) * DG_NP * DG_NP];
  const DG_FP *ds_mat = &dg_Ds_kernel[(DG_ORDER - 1) * DG_NP * DG_NP];

  DG_FP xr = 0.0;
  DG_FP yr = 0.0;
  for(int i = 0; i < DG_NP; i++) {
    int ind = DG_MAT_IND(0, i, DG_NP, DG_NP);
    xr += dr_mat[ind] * x[i];
    yr += dr_mat[ind] * y[i];
  }

  DG_FP xs = 0.0;
  DG_FP ys = 0.0;
  for(int i = 0; i < DG_NP; i++) {
    int ind = DG_MAT_IND(0, i, DG_NP, DG_NP);
    xs += ds_mat[ind] * x[i];
    ys += ds_mat[ind] * y[i];
  }

  // Calculate normals
  // Face 0
  nx[0] = yr;
  ny[0] = -xr;
  // Face 1
  nx[1] = ys - yr;
  ny[1] = xr - xs;
  // Face 2
  nx[2] = -ys;
  ny[2] = xs;

  // J = -xs.*yr + xr.*ys
  geof[J_IND] = -xs * yr + xr * ys;

  // rx = ys./J; sx =-yr./J; ry =-xs./J; sy = xr./J;
  geof[RX_IND] = ys / geof[J_IND];
  geof[SX_IND] = -yr / geof[J_IND];
  geof[RY_IND] = -xs / geof[J_IND];
  geof[SY_IND] = xr / geof[J_IND];

  // Normalise
  for(int i = 0; i < DG_NUM_FACES; i++) {
    sJ[i] = sqrt(nx[i] * nx[i] + ny[i] * ny[i]);
    nx[i] = nx[i] / sJ[i];
    ny[i] = ny[i] / sJ[i];
    // sJ[i] /= 2.0;
    fscale[i] = sJ[i] / geof[J_IND];
  }
}
