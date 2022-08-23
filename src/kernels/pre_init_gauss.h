inline void pre_init_gauss(const int *p, const double *x, const double *y, 
                           const double *gF0Dr_g, const double *gF0Ds_g, 
                           const double *gF1Dr_g, const double *gF1Ds_g, 
                           const double *gF2Dr_g, const double *gF2Ds_g, 
                           double *rx, double *sx, double *ry, double *sy) {
  const int dg_np    = DG_CONSTANTS[(*p - 1) * 5];

  const double *gF0Dr = &gF0Dr_g[(*p - 1) * DG_GF_NP * DG_NP];
  const double *gF0Ds = &gF0Ds_g[(*p - 1) * DG_GF_NP * DG_NP];
  const double *gF1Dr = &gF1Dr_g[(*p - 1) * DG_GF_NP * DG_NP];
  const double *gF1Ds = &gF1Ds_g[(*p - 1) * DG_GF_NP * DG_NP];
  const double *gF2Dr = &gF2Dr_g[(*p - 1) * DG_GF_NP * DG_NP];
  const double *gF2Ds = &gF2Ds_g[(*p - 1) * DG_GF_NP * DG_NP];
  
  for(int m = 0; m < DG_GF_NP; m++) {
    rx[m] = 0.0;
    sx[m] = 0.0;
    ry[m] = 0.0;
    sy[m] = 0.0;
    for(int n = 0; n < dg_np; n++) {
      int ind = m + n * DG_GF_NP;
      rx[m] += gF0Dr[ind] * x[n];
      sx[m] += gF0Ds[ind] * x[n];
      ry[m] += gF0Dr[ind] * y[n];
      sy[m] += gF0Ds[ind] * y[n];
    }
  }

  for(int m = 0; m < DG_GF_NP; m++) {
    rx[m + DG_GF_NP] = 0.0;
    sx[m + DG_GF_NP] = 0.0;
    ry[m + DG_GF_NP] = 0.0;
    sy[m + DG_GF_NP] = 0.0;
    for(int n = 0; n < dg_np; n++) {
      int ind = m + n * DG_GF_NP;
      rx[m + DG_GF_NP] += gF1Dr[ind] * x[n];
      sx[m + DG_GF_NP] += gF1Ds[ind] * x[n];
      ry[m + DG_GF_NP] += gF1Dr[ind] * y[n];
      sy[m + DG_GF_NP] += gF1Ds[ind] * y[n];
    }
  }

  for(int m = 0; m < DG_GF_NP; m++) {
    rx[m + 2 * DG_GF_NP] = 0.0;
    sx[m + 2 * DG_GF_NP] = 0.0;
    ry[m + 2 * DG_GF_NP] = 0.0;
    sy[m + 2 * DG_GF_NP] = 0.0;
    for(int n = 0; n < dg_np; n++) {
      int ind = m + n * DG_GF_NP;
      rx[m + 2 * DG_GF_NP] += gF2Dr[ind] * x[n];
      sx[m + 2 * DG_GF_NP] += gF2Ds[ind] * x[n];
      ry[m + 2 * DG_GF_NP] += gF2Dr[ind] * y[n];
      sy[m + 2 * DG_GF_NP] += gF2Ds[ind] * y[n];
    }
  }
}