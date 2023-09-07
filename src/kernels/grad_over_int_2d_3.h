inline void grad_over_int_2d_3(const DG_FP *nx, const DG_FP *ny, const DG_FP *sJ, 
                               const DG_FP *geof, DG_FP *mU, DG_FP *pU) {
  for(int i = 0; i < DG_NUM_FACES * DG_CUB_SURF_2D_NP; i++) {
    const int face_ind = i/DG_CUB_SURF_2D_NP;
    const DG_FP _nx = nx[face_ind];
    const DG_FP _ny = ny[face_ind];
    const DG_FP _fscale = sJ[face_ind] / geof[J_IND];
    // const DG_FP _fscale = sJ[face_ind];
    const DG_FP _mU = mU[i];
    const DG_FP _pU = pU[i];

    mU[i] = _fscale * _nx * 0.5 * (_mU + _pU);
    pU[i] = _fscale * _ny * 0.5 * (_mU + _pU);
  }
}