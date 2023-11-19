inline void fpmf_2d_mult_flux_oi_1(const int *order, const DG_FP *nx, const DG_FP *ny,
                                   const DG_FP *sJ, const DG_FP *tau,
                                   const DG_FP *factor, DG_FP *jump, DG_FP *avg_x,
                                   DG_FP *avg_y) {
  const int dg_npf = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS + 1];

  for(int i = 0; i < DG_NUM_FACES; i++) {
    for(int j = 0; j < dg_npf; j++) {
      const DG_FP _jump = 0.5 * jump[i * dg_npf + j];
      const DG_FP _sum = nx[i] * avg_x[i * dg_npf + j]
                       + ny[i] * avg_y[i * dg_npf + j];
      jump[i * dg_npf + j]  = sJ[i] * (tau[i] * _jump - _sum);

    }
  }
}
