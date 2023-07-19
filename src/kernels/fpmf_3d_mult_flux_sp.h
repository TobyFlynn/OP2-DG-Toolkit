inline void fpmf_3d_mult_flux_sp(const int *order, const DG_FP *nx, const DG_FP *ny,
                              const DG_FP *nz, const DG_FP *sJ, const DG_FP *tau,
                              const float *factor, float *jump, float *avg_x,
                              float *avg_y, float *avg_z) {
  const int dg_npf = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS + 1];

  for(int i = 0; i < 4; i++) {
    const float _nx = (float)nx[i];
    const float _ny = (float)ny[i];
    const float _nz = (float)nz[i];
    const float _sJ = (float)sJ[i];
    const float _tau = (float)tau[i];
    for(int j = 0; j < dg_npf; j++) {
      const float _jump = 0.5f * jump[i * dg_npf + j];
      const float _sum = _nx * avg_x[i * dg_npf + j]
                       + _ny * avg_y[i * dg_npf + j]
                       + _nz * avg_z[i * dg_npf + j];
      jump[i * dg_npf + j]  = _sJ * (_tau * _jump - _sum);
      const int factor_ind = FMASK_TK[(*order - 1) * 4 * DG_NPF + i * dg_npf + j];
      const DG_FP fact = factor[factor_ind];
      avg_x[i * dg_npf + j] = _nx * _sJ * fact * -_jump;
      avg_y[i * dg_npf + j] = _ny * _sJ * fact * -_jump;
      avg_z[i * dg_npf + j] = _nz * _sJ * fact * -_jump;
    }
  }
}
