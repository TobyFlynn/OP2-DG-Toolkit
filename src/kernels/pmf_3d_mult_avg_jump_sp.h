inline void pmf_3d_mult_avg_jump_sp(const int * __restrict__ order, const int * __restrict__ bc_type,
                               const int * __restrict__ faceNum, const float * __restrict__ in,
                               const float * __restrict__ in_x, const float * __restrict__ in_y,
                               const float * __restrict__ in_z, float * __restrict__ jump, float * __restrict__ avg_x,
                               float * __restrict__ avg_y, float * __restrict__ avg_z) {
  if(*bc_type == 1)
    return;

  const int p = *order;
  const int dg_npf = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS + 1];
  const int _faceNum = *faceNum;

  for(int j = 0; j < dg_npf; j++) {
    const int fmaskInd = FMASK_TK[(p - 1) * 4 * DG_NPF + _faceNum * dg_npf + j];
    const int ind = _faceNum * dg_npf + j;
    jump[ind] += 2.0 * in[fmaskInd];
    avg_x[ind] += in_x[fmaskInd];
    avg_y[ind] += in_y[fmaskInd];
    avg_z[ind] += in_z[fmaskInd];
  }
}
