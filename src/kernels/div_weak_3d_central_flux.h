inline void div_weak_3d_central_flux(const int *faceNum, const int *fmaskL_corrected,
                                     const int *fmaskR_corrected, const DG_FP *nx,
                                     const DG_FP *ny, const DG_FP *nz,
                                     const DG_FP *fscale, const DG_FP **u,
                                     const DG_FP **v, const DG_FP **w,
                                     DG_FP **flux) {
  const int *fmask  = &FMASK_TK[(DG_ORDER - 1) * DG_NUM_FACES * DG_NPF];
  const int *fmaskL = &fmask[faceNum[0] * DG_NPF];
  const int *fmaskR = &fmask[faceNum[1] * DG_NPF];

  for(int i = 0; i < DG_NPF; i++) {
    int find = faceNum[0] * DG_NPF + i;
    DG_FP fluxU = 0.5 * (u[0][fmaskL[i]] + u[1][fmaskR_corrected[i]]);
    DG_FP fluxV = 0.5 * (v[0][fmaskL[i]] + v[1][fmaskR_corrected[i]]);
    DG_FP fluxW = 0.5 * (w[0][fmaskL[i]] + w[1][fmaskR_corrected[i]]);
    flux[0][find] += fscale[0] * (nx[0] * fluxU + ny[0] * fluxV + nz[0] * fluxW);
  }

  for(int i = 0; i < DG_NPF; i++) {
    int find = faceNum[1] * DG_NPF + i;
    DG_FP fluxU = 0.5 * (u[0][fmaskL_corrected[i]] + u[1][fmaskR[i]]);
    DG_FP fluxV = 0.5 * (v[0][fmaskL_corrected[i]] + v[1][fmaskR[i]]);
    DG_FP fluxW = 0.5 * (w[0][fmaskL_corrected[i]] + w[1][fmaskR[i]]);
    flux[1][find] += fscale[1] * (nx[1] * fluxU + ny[1] * fluxV + nz[1] * fluxW);
  }
}