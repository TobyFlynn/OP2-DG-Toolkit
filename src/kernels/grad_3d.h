inline void grad_3d(const double *ur, const double *us, const double *ut,
                    const double *rx, const double *sx, const double *tx,
                    const double *ry, const double *sy, const double *ty,
                    const double *rz, const double *sz, const double *tz,
                    double *ux, double *uy, double *uz) {
  for(int i = 0; i < DG_NP; i++) {
    ux[i] = *rx * ur[i] + *sx * us[i] + *tx * ut[i];
    uy[i] = *ry * ur[i] + *sy * us[i] + *ty * ut[i];
    uz[i] = *rz * ur[i] + *sz * us[i] + *tz * ut[i];
  }
}