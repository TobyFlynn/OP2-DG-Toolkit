inline void grad_3d(const DG_FP *ur, const DG_FP *us, const DG_FP *ut,
                    const DG_FP *rx, const DG_FP *sx, const DG_FP *tx,
                    const DG_FP *ry, const DG_FP *sy, const DG_FP *ty,
                    const DG_FP *rz, const DG_FP *sz, const DG_FP *tz,
                    DG_FP *ux, DG_FP *uy, DG_FP *uz) {
  for(int i = 0; i < DG_NP; i++) {
    ux[i] = *rx * ur[i] + *sx * us[i] + *tx * ut[i];
    uy[i] = *ry * ur[i] + *sy * us[i] + *ty * ut[i];
    uz[i] = *rz * ur[i] + *sz * us[i] + *tz * ut[i];
  }
}