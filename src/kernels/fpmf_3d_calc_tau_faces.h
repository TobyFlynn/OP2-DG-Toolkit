inline void fpmf_3d_calc_tau_faces(const int * __restrict__ order, const int * __restrict__ faceNum,
                                   const int * __restrict__ fmaskR_corrected,
                                   const DG_FP * __restrict__ fscale, const DG_FP ** __restrict__ factor,
                                   DG_FP ** __restrict__ tau) {
  const int p = *order;
  const int dg_npf = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS + 1];
  DG_FP gtau = 0.0;
  const int faceNumL = faceNum[0];
  for(int j = 0; j < dg_npf; j++) {
    const int fmaskL_ind = FMASK_TK[(p - 1) * 4 * DG_NPF + faceNumL * dg_npf + j];
    const int fmaskR_ind = fmaskR_corrected[j];
    DG_FP tmp = 2.0 * (p + 1) * (p + 2) * fmax(fscale[0] * factor[0][fmaskL_ind], fscale[1] * factor[1][fmaskR_ind]);
    gtau = fmax(gtau, tmp);
  }
  const int faceNumR = faceNum[1];
  tau[0][faceNumL] = gtau;
  tau[1][faceNumR] = gtau;
}
