inline void grad_2d_central_flux(const int *faceNum, const bool *reverse,
                                 const DG_FP *nx, const DG_FP *ny,
                                 const DG_FP *fscale, const DG_FP **u,
                                 DG_FP **fluxX, DG_FP **fluxY) {
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

    DG_FP flux = u[0][fmaskL_ind] - 0.5 * (u[0][fmaskL_ind] + u[1][fmaskR_ind]);
    fluxX[0][findL] = fscale[0] * nx[0] * flux;
    fluxY[0][findL] = fscale[0] * ny[0] * flux;
    fluxX[1][findR] = fscale[1] * nx[1] * flux;
    fluxY[1][findR] = fscale[1] * ny[1] * flux;
  }
}
