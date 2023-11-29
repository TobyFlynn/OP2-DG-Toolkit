inline void fpmf_3d_calc_tau_bfaces(const int * __restrict__ order, const int * __restrict__ faceNum,
                                    const DG_FP * __restrict__ fscale, const DG_FP * __restrict__ factor,
                                    DG_FP * __restrict__ tau) {
  const int p = order[0];
  const int dg_npf = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS + 1];
  const int _faceNum = faceNum[0];
  const int fmask_ind_0 = FMASK_TK[(p - 1) * 4 * DG_NPF + _faceNum * dg_npf];
  DG_FP gtau = factor[fmask_ind_0];
  for(int i = 1; i < dg_npf; i++) {
    const int fmask_ind = FMASK_TK[(p - 1) * 4 * DG_NPF + _faceNum * dg_npf + i];
    gtau = fmax(gtau, factor[fmask_ind]);
  }
  gtau *= 2.0 * (p + 1) * (p + 2) * fscale[0];
  tau[_faceNum] = gtau;
}
