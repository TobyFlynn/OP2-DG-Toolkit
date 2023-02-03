inline void div_3d_central_flux(const int *faceNum, const int *fmaskL_corrected,
                                const int *fmaskR_corrected, const double *nx,
                                const double *ny, const double *nz,
                                const double *fscale, const double **u,
                                const double **v, const double **w,
                                double **flux) {
  const int *fmask  = &FMASK[(DG_ORDER - 1) * DG_NUM_FACES * DG_NPF];
  const int *fmaskL = &fmask[faceNum[0] * DG_NPF];
  const int *fmaskR = &fmask[faceNum[1] * DG_NPF];

  for(int i = 0; i < DG_NPF; i++) {
    int find = faceNum[0] * DG_NPF + i;
    double fluxU = u[0][fmaskL[i]] - 0.5 * (u[0][fmaskL[i]] + u[1][fmaskR_corrected[i]]);
    double fluxV = v[0][fmaskL[i]] - 0.5 * (v[0][fmaskL[i]] + v[1][fmaskR_corrected[i]]);
    double fluxW = w[0][fmaskL[i]] - 0.5 * (w[0][fmaskL[i]] + w[1][fmaskR_corrected[i]]);
    flux[0][find] += fscale[0] * (nx[0] * fluxU + ny[0] * fluxV + nz[0] * fluxW);
  }

  for(int i = 0; i < DG_NPF; i++) {
    int find = faceNum[1] * DG_NPF + i;
    double fluxU = u[1][fmaskR[i]] - 0.5 * (u[0][fmaskL_corrected[i]] + u[1][fmaskR[i]]);
    double fluxV = v[1][fmaskR[i]] - 0.5 * (v[0][fmaskL_corrected[i]] + v[1][fmaskR[i]]);
    double fluxW = w[1][fmaskR[i]] - 0.5 * (w[0][fmaskL_corrected[i]] + w[1][fmaskR[i]]);
    flux[1][find] += fscale[1] * (nx[1] * fluxU + ny[1] * fluxV + nz[1] * fluxW);
  }
}