inline void pmf_3d_mult_flux_sp(const int * __restrict__ order, const DG_FP * __restrict__ nx, const DG_FP * __restrict__ ny,
                             const DG_FP * __restrict__ nz, const DG_FP * __restrict__ sJ, const float * __restrict__ tau,
                             float * __restrict__ jump, float * __restrict__ avg_x, float * __restrict__ avg_y,
                             float * __restrict__ avg_z) {
  const int dg_npf = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS + 1];

  for(int i = 0; i < 4; i++) {
    const float _nx = (float)nx[i];
    const float _ny = (float)ny[i];
    const float _nz = (float)nz[i];
    const float _sJ = (float)sJ[i];
    for(int j = 0; j < dg_npf; j++) {
      const float _jump = 0.5 * jump[i * dg_npf + j];
      const float _sum = _nx * avg_x[i * dg_npf + j]
                       + _ny * avg_y[i * dg_npf + j]
                       + _nz * avg_z[i * dg_npf + j];
      jump[i * dg_npf + j]  = _sJ * (tau[i] * _jump - _sum);
      avg_x[i * dg_npf + j] = _nx * _sJ * -_jump;
      avg_y[i * dg_npf + j] = _ny * _sJ * -_jump;
      avg_z[i * dg_npf + j] = _nz * _sJ * -_jump;
    }
  }
}
