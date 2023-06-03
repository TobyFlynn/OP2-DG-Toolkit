inline void div_2d_central_flux(const int *faceNum, const bool *reverse,
                                const DG_FP *nx, const DG_FP *ny,
                                const DG_FP *fscale, const DG_FP **u,
                                const DG_FP **v, DG_FP **flux) {
  const int *fmask  = &FMASK_TK[(DG_ORDER - 1) * DG_NUM_FACES * DG_NPF];
  const int *fmaskL = &fmask[faceNum[0] * DG_NPF];
  const int *fmaskR = &fmask[faceNum[1] * DG_NPF];
  const bool rev = *reverse;

  for(int i = 0; i < DG_NPF; i++) {
    int findL = faceNum[0] * DG_NPF + i;
    int fmaskL_ind = fmaskL[i];
    int findR, fmaskR_ind;
    if(rev) {
      findR = faceNum[1] * DG_NPF + DG_NPF - i - 1;
      fmaskR_ind = fmaskR[DG_NPF - i - 1];
    } else {
      findR = faceNum[1] * DG_NPF + i;
      fmaskR_ind = fmaskR[i];
    }

    const DG_FP u_avg = 0.5 * (u[0][fmaskL_ind] + u[1][fmaskR_ind]);
    const DG_FP fluxUL = u[0][fmaskL_ind] - u_avg;
    const DG_FP fluxUR = u[1][fmaskR_ind] - u_avg;
    const DG_FP v_avg = 0.5 * (v[0][fmaskL_ind] + v[1][fmaskR_ind]);
    const DG_FP fluxVL = v[0][fmaskL_ind] - v_avg;
    const DG_FP fluxVR = v[1][fmaskR_ind] - v_avg;

    flux[0][findL] += fscale[0] * (nx[0] * fluxUL + ny[0] * fluxVL);
    flux[1][findR] += fscale[1] * (nx[1] * fluxUR + ny[1] * fluxVR);
  }
}
