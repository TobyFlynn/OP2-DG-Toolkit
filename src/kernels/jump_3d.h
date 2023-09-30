inline void jump_3d(const int *order, const int *faceNum, const int *fmaskL_corrected,
                    const int *fmaskR_corrected, const DG_FP **in, DG_FP **out) {
  const int p = *order;
  const int dg_npf = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS + 1];
  const int faceNumL = faceNum[0];
  const int faceNumR = faceNum[1];

  for(int j = 0; j < dg_npf; j++) {
    const int fmaskL_ind = FMASK_TK[(p - 1) * 4 * DG_NPF + faceNumL * dg_npf + j];
    const int fmaskR_ind_corr = fmaskR_corrected[j];
    out[0][faceNumL * dg_npf + j] = in[0][fmaskL_ind] - in[1][fmaskR_ind_corr];

    const int fmaskR_ind = FMASK_TK[(p - 1) * 4 * DG_NPF + faceNumR * dg_npf + j];
    const int fmaskL_ind_corr = fmaskL_corrected[j];
    out[1][faceNumR * dg_npf + j] = in[1][fmaskR_ind] - in[0][fmaskL_ind_corr];
  }
}
