inline void grad_3d_central_flux(const int *faceNum, const int *fmaskL_corrected,
                                 const int *fmaskR_corrected, const DG_FP *nx,
                                 const DG_FP *ny, const DG_FP *nz,
                                 const DG_FP *fscale, const DG_FP **u,
                                 DG_FP **fluxX, DG_FP **fluxY, DG_FP **fluxZ) {
  const int *fmask  = &FMASK[(DG_ORDER - 1) * DG_NUM_FACES * DG_NPF];
  const int *fmaskL = &fmask[faceNum[0] * DG_NPF];
  const int *fmaskR = &fmask[faceNum[1] * DG_NPF];

  for(int i = 0; i < DG_NPF; i++) {
    int find = faceNum[0] * DG_NPF + i;
    DG_FP flux = u[0][fmaskL[i]] - 0.5 * (u[0][fmaskL[i]] + u[1][fmaskR_corrected[i]]);
    fluxX[0][find] += fscale[0] * nx[0] * flux;
    fluxY[0][find] += fscale[0] * ny[0] * flux;
    fluxZ[0][find] += fscale[0] * nz[0] * flux;
  }

  for(int i = 0; i < DG_NPF; i++) {
    int find = faceNum[1] * DG_NPF + i;
    DG_FP flux = u[1][fmaskR[i]] - 0.5 * (u[0][fmaskL_corrected[i]] + u[1][fmaskR[i]]);
    fluxX[1][find] += fscale[1] * nx[1] * flux;
    fluxY[1][find] += fscale[1] * ny[1] * flux;
    fluxZ[1][find] += fscale[1] * nz[1] * flux;
  }
}