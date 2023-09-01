inline void fpmf_2d_mult_flux_sp(const int *order, const DG_FP *nx, const DG_FP *ny,
                                 const DG_FP *sJ, const float *tau, const DG_FP *factor, 
                                 float *jump, float *avg_x, float *avg_y) {
  const int dg_npf = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS + 1];

  for(int i = 0; i < DG_NUM_FACES; i++) {
    const float _nx = (float)nx[i];
    const float _ny = (float)ny[i];
    const float _sJ = (float)sJ[i];
    for(int j = 0; j < dg_npf; j++) {
      const float _jump = 0.5f * jump[i * dg_npf + j];
      const float _sum = _nx * avg_x[i * dg_npf + j]
                       + _ny * avg_y[i * dg_npf + j];
      jump[i * dg_npf + j]  = _sJ * (tau[i] * _jump - _sum);
      const int factor_ind = FMASK_TK[(*order - 1) * DG_NUM_FACES * DG_NPF + i * dg_npf + j];
      const float fact = (float)factor[factor_ind];
      avg_x[i * dg_npf + j] = _nx * _sJ * fact * -_jump;
      avg_y[i * dg_npf + j] = _ny * _sJ * fact * -_jump;
    }
  }
}
