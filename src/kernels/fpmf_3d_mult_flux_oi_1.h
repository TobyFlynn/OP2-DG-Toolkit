inline void fpmf_3d_mult_flux_oi_1(const int *order, const DG_FP *nx, const DG_FP *ny,
                                   const DG_FP *nz, const DG_FP *sJ, const DG_FP *tau,
                                   DG_FP *jump, DG_FP *avg_x, DG_FP *avg_y, DG_FP *avg_z) {
  const int dg_npf = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS + 1];

  for(int i = 0; i < DG_NUM_FACES; i++) {
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
      jump[i * dg_npf + j] = _sJ * (_tau * _jump - _sum);
    }
  }
}
