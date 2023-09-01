inline void fpmf_2d_calc_tau_faces_sp(const int *order, const int *faceNum,
                                   const bool *reverse, const DG_FP *fscale,
                                   const DG_FP **factor, float **tau) {
  const int p = *order;
  const int dg_npf = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS + 1];
  DG_FP gtau = 0.0;
  const int faceNumL = faceNum[0];
  const int faceNumR = faceNum[1];
  for(int j = 0; j < dg_npf; j++) {
    const int fmaskL_ind = FMASK_TK[(p - 1) * DG_NUM_FACES * DG_NPF + faceNumL * dg_npf + j];
    const int fmaskR_ind = *reverse ? FMASK_TK[(p - 1) * DG_NUM_FACES * DG_NPF + faceNumR * dg_npf + dg_npf - j - 1] : FMASK_TK[(p - 1) * DG_NUM_FACES * DG_NPF + faceNumR * dg_npf + j];
    DG_FP tmp = 2.0 * (p + 1) * (p + 2) * fmax(fscale[0] * factor[0][fmaskL_ind], fscale[1] * factor[1][fmaskR_ind]);
    gtau = fmax(gtau, tmp);
  }
  tau[0][faceNumL] = (float)gtau;
  tau[1][faceNumR] = (float)gtau;
}
