inline void grad_3d_central_flux(const int *faceNum, const int *fmaskL_corrected,
                                 const int *fmaskR_corrected, const double *nx,
                                 const double *ny, const double *nz,
                                 const double *fscale, const double **u,
                                 double **fluxX, double **fluxY, double **fluxZ) {
  const int *fmask  = &FMASK[(DG_ORDER - 1) * DG_NUM_FACES * DG_NPF];
  const int *fmaskL = &fmask[faceNum[0] * DG_NPF];
  const int *fmaskR = &fmask[faceNum[1] * DG_NPF];

  for(int i = 0; i < DG_NPF; i++) {
    int find = faceNum[0] * DG_NPF + i;
    double flux = u[0][fmaskL[i]] - 0.5 * (u[0][fmaskL[i]] + u[1][fmaskR_corrected[i]]);
    fluxX[0][find] += fscale[0] * nx[0] * flux;
    fluxY[0][find] += fscale[0] * ny[0] * flux;
    fluxZ[0][find] += fscale[0] * nz[0] * flux;
  }

  for(int i = 0; i < DG_NPF; i++) {
    int find = faceNum[1] * DG_NPF + i;
    double flux = u[1][fmaskR[i]] - 0.5 * (u[0][fmaskL_corrected[i]] + u[1][fmaskR[i]]);
    fluxX[1][find] += fscale[1] * nx[1] * flux;
    fluxY[1][find] += fscale[1] * ny[1] * flux;
    fluxZ[1][find] += fscale[1] * nz[1] * flux;
  }
}