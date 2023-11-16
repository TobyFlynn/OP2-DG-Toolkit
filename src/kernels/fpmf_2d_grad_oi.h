inline void fpmf_2d_grad_oi(const int *p, const DG_FP *geof,
                         const DG_FP * __restrict__ fact,
                         DG_FP * __restrict__ ux, DG_FP * __restrict__ uy) {
  for(int m = 0; m < DG_CUB_2D_NP; m++) {
    const DG_FP r = ux[m];
    const DG_FP s = uy[m];
    ux[m] = fact[m] * (geof[RX_IND] * r + geof[SX_IND] * s);
    uy[m] = fact[m] * (geof[RY_IND] * r + geof[SY_IND] * s);
  }
}
