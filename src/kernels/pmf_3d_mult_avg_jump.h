inline void pmf_3d_mult_avg_jump(const int * __restrict__ order, const int * __restrict__ bc_type,
                               const int * __restrict__ faceNum, const DG_FP * __restrict__ in,
                               const DG_FP * __restrict__ in_x, const DG_FP * __restrict__ in_y,
                               const DG_FP * __restrict__ in_z, DG_FP * __restrict__ jump, DG_FP * __restrict__ avg_x,
                               DG_FP * __restrict__ avg_y, DG_FP * __restrict__ avg_z) {
  if(*bc_type == 1)
    return;

  const int p = order[0];
  const int dg_npf = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS + 1];

  for(int j = 0; j < dg_npf; j++) {
    const int fmaskInd = FMASK_TK[(p - 1) * 4 * DG_NPF + faceNum[0] * dg_npf + j];
    const int ind = faceNum[0] * dg_npf + j;
    jump[ind] += 2.0 * in[fmaskInd];
    avg_x[ind] += in_x[fmaskInd];
    avg_y[ind] += in_y[fmaskInd];
    avg_z[ind] += in_z[fmaskInd];
  }
}
