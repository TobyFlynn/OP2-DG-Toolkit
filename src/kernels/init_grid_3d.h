inline void init_grid_3d(const int *order, const DG_FP *cX, const DG_FP *cY,
                         const DG_FP *cZ, DG_FP *x, DG_FP *y, DG_FP *z) {
  const int dg_np = DG_CONSTANTS_TK[(*order - 1) * 2];
  const DG_FP *r = dg_r_kernel + (*order - 1) * DG_NP;
  const DG_FP *s = dg_s_kernel + (*order - 1) * DG_NP;
  const DG_FP *t = dg_t_kernel + (*order - 1) * DG_NP;
  for(int i = 0; i < dg_np; i++) {
    x[i] = 0.5 * (-(1.0 + r[i] + s[i] + t[i]) * cX[0] + (1.0 + r[i]) * cX[1] + (1.0 + s[i]) * cX[2] + (1.0 + t[i]) * cX[3]);
    y[i] = 0.5 * (-(1.0 + r[i] + s[i] + t[i]) * cY[0] + (1.0 + r[i]) * cY[1] + (1.0 + s[i]) * cY[2] + (1.0 + t[i]) * cY[3]);
    z[i] = 0.5 * (-(1.0 + r[i] + s[i] + t[i]) * cZ[0] + (1.0 + r[i]) * cZ[1] + (1.0 + s[i]) * cZ[2] + (1.0 + t[i]) * cZ[3]);
  }
}
