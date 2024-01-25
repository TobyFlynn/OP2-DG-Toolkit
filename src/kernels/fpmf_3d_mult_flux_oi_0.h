inline void fpmf_3d_mult_flux_oi_0(const DG_FP *nx, const DG_FP *ny, const DG_FP *nz,
                                   const DG_FP *sJ, const DG_FP *geof, const DG_FP *factor,
                                   const DG_FP *jump, DG_FP *out_x, DG_FP *out_y, DG_FP *out_z) {
  for(int i = 0; i < DG_NUM_FACES * DG_CUB_SURF_3D_NP; i++) {
    const int face_ind = i/DG_CUB_SURF_3D_NP;
    const DG_FP _nx = nx[face_ind];
    const DG_FP _ny = ny[face_ind];
    const DG_FP _nz = nz[face_ind];
    const DG_FP _fscale = sJ[face_ind] / geof[J_IND];

    out_x[i] = _nx * _fscale * factor[i] * 0.5 * -jump[i];
    out_y[i] = _ny * _fscale * factor[i] * 0.5 * -jump[i];
    out_z[i] = _nz * _fscale * factor[i] * 0.5 * -jump[i];
  }
}
