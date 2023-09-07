inline void grad_over_int_2d_1(const int *faceNum, const bool *reverse,
                               const DG_FP **u, DG_FP **uM, DG_FP **uP) {
  const bool rev = *reverse;
  const int edgeL = faceNum[0];
  const int edgeR = faceNum[1];
  const int *fmaskL = &FMASK_TK[(DG_ORDER - 1) * DG_NUM_FACES * DG_NPF + edgeL * DG_NPF];
  const int *fmaskR = &FMASK_TK[(DG_ORDER - 1) * DG_NUM_FACES * DG_NPF + edgeR * DG_NPF];

  for(int i = 0; i < DG_NPF; i++) {
    const int indL = fmaskL[i];
    const int indR = rev ? fmaskR[DG_NPF - i - 1] : fmaskR[i];
    
    uM[0][edgeL * DG_NPF + i] = u[0][indL];
    uP[0][edgeL * DG_NPF + i] = u[1][indR];
  }

  for(int i = 0; i < DG_NPF; i++) {
    const int indR = fmaskR[i];
    const int indL = rev ? fmaskL[DG_NPF - i - 1] : fmaskL[i];
    
    uM[1][edgeR * DG_NPF + i] = u[1][indR];
    uP[1][edgeR * DG_NPF + i] = u[0][indL];
  }
}