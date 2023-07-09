inline void avg_2d(const int *order, const int *faceNum, const bool *reverse,
                   const DG_FP **in, DG_FP **out) {
  const int p = *order;
  const int dg_npf = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS + 1];
  const int faceNumL = faceNum[0];
  const int faceNumR = faceNum[1];

  for(int j = 0; j < dg_npf; j++) {
    const int fmaskL_ind = FMASK_TK[(p - 1) * DG_NUM_FACES * DG_NPF + faceNumL * dg_npf + j];
    const int fmaskR_ind_corr = *reverse ? FMASK_TK[(p - 1) * DG_NUM_FACES * DG_NPF + faceNumR * dg_npf + dg_npf - j - 1] : FMASK_TK[(p - 1) * DG_NUM_FACES * DG_NPF + faceNumR * dg_npf + j];
    const DG_FP avg = 0.5 * (in[0][fmaskL_ind] + in[1][fmaskR_ind_corr]);
    out[0][faceNumL * dg_npf + j] = avg;
    const int r_out_ind = *reverse ? faceNumR * dg_npf + dg_npf - j - 1 : faceNumR * dg_npf + j;
    out[1][r_out_ind] = avg;
  }
}
