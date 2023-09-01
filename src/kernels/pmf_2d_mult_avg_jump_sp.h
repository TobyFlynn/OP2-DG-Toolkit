inline void pmf_2d_mult_avg_jump_sp(const int *order, const int *bc_type,
                              const int *faceNum, const float *in,
                              const float *in_x, const float *in_y,
                              float *jump, float *avg_x, float *avg_y) {
  if(*bc_type == 1)
    return;

  const int p = order[0];
  const int dg_npf = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS + 1];

  for(int j = 0; j < dg_npf; j++) {
    const int fmaskInd = FMASK_TK[(p - 1) * DG_NUM_FACES * DG_NPF + faceNum[0] * dg_npf + j];
    const int ind = faceNum[0] * dg_npf + j;
    jump[ind] += 2.0 * in[fmaskInd];
    avg_x[ind] += in_x[fmaskInd];
    avg_y[ind] += in_y[fmaskInd];
  }
}
