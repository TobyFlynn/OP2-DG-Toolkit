inline void fpmf_3d_mult_flux_oi_sp_0(const DG_FP *nx, const DG_FP *ny, const DG_FP *nz,
                                   const DG_FP *sJ, const DG_FP *geof, const DG_FP *factor,
                                   const float *jump, float *out_x, float *out_y, float *out_z) {
  for(int i = 0; i < DG_NUM_FACES * DG_CUB_SURF_3D_NP; i++) {
    const int face_ind = i/DG_CUB_SURF_3D_NP;
    const float _nx = (float)nx[face_ind];
    const float _ny = (float)ny[face_ind];
    const float _nz = (float)nz[face_ind];
    const float _fscale = (float)sJ[face_ind] / (float)geof[J_IND];
    const float _factor = (float)factor[i];

    out_x[i] = _nx * _fscale * _factor * 0.5f * -jump[i];
    out_y[i] = _ny * _fscale * _factor * 0.5f * -jump[i];
    out_z[i] = _nz * _fscale * _factor * 0.5f * -jump[i];
  }
}
