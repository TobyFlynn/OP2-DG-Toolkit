inline void flux_init_3d(const int *p, const int **faceNum, const int **fmaskL,
                         const int **fmaskR, const DG_FP **nx, const DG_FP **ny,
                         const DG_FP **nz, const DG_FP **sJ, const DG_FP **fscale,
                         const int *fluxL, int *fluxFaceNums, int *fluxFmask,
                         DG_FP *fluxNx, DG_FP *fluxNy, DG_FP *fluxNz, DG_FP *fluxSJ,
                         DG_FP *fluxFscale) {
  const int dg_np  = DG_CONSTANTS_TK[(*p - 1) * 2];
  const int dg_npf = DG_CONSTANTS_TK[(*p - 1) * 2 + 1];

  // Set face numbers
  for(int i = 0; i < DG_NUM_FACES; i++) {
    if(fluxL[i]) {
      fluxFaceNums[2 * i] = faceNum[i][0];
      fluxFaceNums[2 * i + 1] = faceNum[i][1];
    } else {
      fluxFaceNums[2 * i] = faceNum[i][1];
      fluxFaceNums[2 * i + 1] = faceNum[i][0];
    }
  }

  // Set fmask
  for(int i = 0; i < DG_NUM_FACES; i++) {
    const int ind = i * dg_npf;
    if(fluxL[i]) {
      for(int j = 0; j < dg_npf; j++) {
        fluxFmask[ind + j] = fmaskR[i][j];
      }
    } else {
      for(int j = 0; j < dg_npf; j++) {
        fluxFmask[ind + j] = fmaskL[i][j];
      }
    }
  }

  // Set normals
  for(int i = 0; i < DG_NUM_FACES; i++) {
    if(fluxL[i]) {
      fluxNx[i] = nx[i][0];
      fluxNy[i] = ny[i][0];
      fluxNz[i] = nz[i][0];
      fluxSJ[i] = sJ[i][0];
      fluxFscale[2 * i] = fscale[i][0];
      fluxFscale[2 * i + 1] = fscale[i][1];
    } else {
      fluxNx[i] = nx[i][1];
      fluxNy[i] = ny[i][1];
      fluxNz[i] = nz[i][1];
      fluxSJ[i] = sJ[i][1];
      fluxFscale[2 * i] = fscale[i][1];
      fluxFscale[2 * i + 1] = fscale[i][0];
    }
  }
}
