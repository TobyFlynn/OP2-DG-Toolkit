inline void fpmf_3d_mult_flux(const int *order, const DG_FP *nx, const DG_FP *ny,
                              const DG_FP *nz, const DG_FP *sJ, const DG_FP *tau,
                              const DG_FP *factor, DG_FP *jump, DG_FP *avg_x,
                              DG_FP *avg_y, DG_FP *avg_z) {
  const int p = *order;
  const int dg_npf = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS + 1];

  for(int i = 0; i < 4; i++) {
    const DG_FP _nx = nx[i];
    const DG_FP _ny = ny[i];
    const DG_FP _nz = nz[i];
    const DG_FP _sJ = sJ[i];
    const DG_FP _tau = tau[i];
    for(int j = 0; j < dg_npf; j++) {
      const DG_FP _jump = 0.5 * jump[i * dg_npf + j];
      const DG_FP _sum = _nx * avg_x[i * dg_npf + j]
                       + _ny * avg_y[i * dg_npf + j]
                       + _nz * avg_z[i * dg_npf + j];
      jump[i * dg_npf + j]  = _sJ * (_tau * _jump - _sum);
      const int factor_ind = FMASK_TK[(p - 1) * 4 * DG_NPF + i * dg_npf + j];
      const DG_FP fact = factor[factor_ind];
      avg_x[i * dg_npf + j] = _nx * _sJ * fact * -_jump;
      avg_y[i * dg_npf + j] = _ny * _sJ * fact * -_jump;
      avg_z[i * dg_npf + j] = _nz * _sJ * fact * -_jump;
    }
  }
}
