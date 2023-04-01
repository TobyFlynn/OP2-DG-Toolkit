inline void grad_3d_geof(const int *p, const DG_FP *rx, const DG_FP *sx,
                    const DG_FP *tx, const DG_FP *ry, const DG_FP *sy,
                    const DG_FP *ty, const DG_FP *rz, const DG_FP *sz,
                    const DG_FP *tz, const DG_FP *ur, const DG_FP *us,
                    const DG_FP *ut, DG_FP *ux, DG_FP *uy, DG_FP *uz) {
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  for(int m = 0; m < dg_np; m++) {
    ux[m] = *rx * ur[m] + *sx * us[m] + *tx * ut[m];
    uy[m] = *ry * ur[m] + *sy * us[m] + *ty * ut[m];
    uz[m] = *rz * ur[m] + *sz * us[m] + *tz * ut[m];
  }
}
