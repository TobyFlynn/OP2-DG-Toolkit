inline void fpmf_3d_mult_flux_sp(const int * __restrict__ order, const DG_FP * __restrict__ nx, const DG_FP * __restrict__ ny,
                              const DG_FP * __restrict__ nz, const DG_FP * __restrict__ sJ, const DG_FP * __restrict__ tau,
                              const float * __restrict__ factor, float * __restrict__ jump, float * __restrict__ avg_x,
                              float * __restrict__ avg_y, float * __restrict__ avg_z) {
  const int p = *order;
  const int dg_npf = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS + 1];

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
      const int factor_ind = FMASK_TK[(p - 1) * 4 * DG_NPF + i * dg_npf + j];
      const float fact = factor[factor_ind];
      avg_x[i * dg_npf + j] = _nx * _sJ * fact * -_jump;
      avg_y[i * dg_npf + j] = _ny * _sJ * fact * -_jump;
      avg_z[i * dg_npf + j] = _nz * _sJ * fact * -_jump;
    }
  }
}
