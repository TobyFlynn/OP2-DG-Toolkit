inline void grad_2d_central_flux(const int *edgeNum, const bool *rev,
                                 const double **nx, const double **ny,
                                 const double **sJ, const double **u, 
                                 double **ux, double **uy) {
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
    double flux = u[0][lInd] - 0.5 * (u[0][lInd] + u[1][rInd]);
    ux[0][lInd] += gaussW_g[i] * sJ[0][lInd] * nx[0][lInd] * flux;
    uy[0][lInd] += gaussW_g[i] * sJ[0][lInd] * ny[0][lInd] * flux;
  }

  for(int i = 0; i < DG_GF_NP; i++) {
    int rInd = exIndR + i;
    int lInd;
    if(reverse) {
      lInd = exIndL + DG_GF_NP - 1 - i;
    } else {
      lInd = exIndL + i;
    }
    double flux = u[1][rInd] - 0.5 * (u[0][lInd] + u[1][rInd]);
    ux[1][rInd] += gaussW_g[i] * sJ[1][rInd] * nx[1][rInd] * flux;
    uy[1][rInd] += gaussW_g[i] * sJ[1][rInd] * ny[1][rInd] * flux;
  }
}