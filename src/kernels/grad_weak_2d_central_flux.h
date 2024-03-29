inline void grad_weak_2d_central_flux(const int *edgeNum, const bool *rev,
                                      const DG_FP **nx, const DG_FP **ny,
                                      const DG_FP **sJ, const DG_FP **u, 
                                      DG_FP **ux, DG_FP **uy) {
  // Work out which edge for each element
  int edgeL = edgeNum[0];
  int edgeR = edgeNum[1];
  bool reverse = *rev;

  // Copy data from R to L
  int exIndL = edgeL * DG_GF_NP;
  int exIndR = edgeR * DG_GF_NP;

  for(int i = 0; i < DG_GF_NP; i++) {
    int lInd = exIndL + i;
    int rInd;
    if(reverse) {
      rInd = exIndR + DG_GF_NP - 1 - i;
    } else {
      rInd = exIndR + i;
    }
    DG_FP flux = 0.5 * (u[0][lInd] + u[1][rInd]);
    ux[0][lInd] += gaussW_g_TK[i] * sJ[0][lInd] * nx[0][lInd] * flux;
    uy[0][lInd] += gaussW_g_TK[i] * sJ[0][lInd] * ny[0][lInd] * flux;
  }

  for(int i = 0; i < DG_GF_NP; i++) {
    int rInd = exIndR + i;
    int lInd;
    if(reverse) {
      lInd = exIndL + DG_GF_NP - 1 - i;
    } else {
      lInd = exIndL + i;
    }
    DG_FP flux = 0.5 * (u[0][lInd] + u[1][rInd]);
    ux[1][rInd] += gaussW_g_TK[i] * sJ[1][rInd] * nx[1][rInd] * flux;
    uy[1][rInd] += gaussW_g_TK[i] * sJ[1][rInd] * ny[1][rInd] * flux;
  }
}
