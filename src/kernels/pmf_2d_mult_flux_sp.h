inline void pmf_2d_mult_flux_sp(const int *order, const DG_FP *nx, const DG_FP *ny,
                                const DG_FP *sJ, const float *tau, float *jump,
                                float *avg_x, float *avg_y) {
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
      avg_x[i * dg_npf + j] = _nx * _sJ * -_jump;
      avg_y[i * dg_npf + j] = _ny * _sJ * -_jump;
    }
  }
}
