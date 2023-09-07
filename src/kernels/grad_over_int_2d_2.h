inline void grad_over_int_2d_2(const int *bedgeNum, const DG_FP *u, 
                               DG_FP *uM, DG_FP *uP) {
  const int edge = bedgeNum[0];
  const int *fmask = &FMASK_TK[(DG_ORDER - 1) * DG_NUM_FACES * DG_NPF + edge * DG_NPF];

  // Simple for surface tension case
  for(int i = 0; i < DG_NPF; i++) {
    const int find = fmask[i];
    
    uM[edge * DG_NPF + i] = u[find];
    uP[edge * DG_NPF + i] = u[find];
  }
}