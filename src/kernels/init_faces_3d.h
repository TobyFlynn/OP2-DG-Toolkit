inline void init_faces_3d(const int *faceNum, const DG_FP **geof, DG_FP *nx,
                          DG_FP *ny, DG_FP *nz, DG_FP *sJ, DG_FP *fscale) {
  if(faceNum[0] == 0) {
    nx[0] = -geof[0][TX_IND];
    ny[0] = -geof[0][TY_IND];
    nz[0] = -geof[0][TZ_IND];
  } else if(faceNum[0] == 1) {
    nx[0] = -geof[0][SX_IND];
    ny[0] = -geof[0][SY_IND];
    nz[0] = -geof[0][SZ_IND];
  } else if(faceNum[0] == 2) {
    nx[0] = geof[0][RX_IND] + geof[0][SX_IND] + geof[0][TX_IND];
    ny[0] = geof[0][RY_IND] + geof[0][SY_IND] + geof[0][TY_IND];
    nz[0] = geof[0][RZ_IND] + geof[0][SZ_IND] + geof[0][TZ_IND];
  } else {
    nx[0] = -geof[0][RX_IND];
    ny[0] = -geof[0][RY_IND];
    nz[0] = -geof[0][RZ_IND];
  }

  if(faceNum[1] == 0) {
    nx[1] = -geof[1][TX_IND];
    ny[1] = -geof[1][TY_IND];
    nz[1] = -geof[1][TZ_IND];
  } else if(faceNum[1] == 1) {
    nx[1] = -geof[1][SX_IND];
    ny[1] = -geof[1][SY_IND];
    nz[1] = -geof[1][SZ_IND];
  } else if(faceNum[1] == 2) {
    nx[1] = geof[1][RX_IND] + geof[1][SX_IND] + geof[1][TX_IND];
    ny[1] = geof[1][RY_IND] + geof[1][SY_IND] + geof[1][TY_IND];
    nz[1] = geof[1][RZ_IND] + geof[1][SZ_IND] + geof[1][TZ_IND];
  } else {
    nx[1] = -geof[1][RX_IND];
    ny[1] = -geof[1][RY_IND];
    nz[1] = -geof[1][RZ_IND];
  }

  sJ[0] = sqrt(nx[0] * nx[0] + ny[0] * ny[0] + nz[0] * nz[0]);
  sJ[1] = sqrt(nx[1] * nx[1] + ny[1] * ny[1] + nz[1] * nz[1]);
  nx[0] /= sJ[0];
  ny[0] /= sJ[0];
  nz[0] /= sJ[0];
  nx[1] /= sJ[1];
  ny[1] /= sJ[1];
  nz[1] /= sJ[1];
  sJ[0] *= geof[0][J_IND];
  fscale[0] = sJ[0] / geof[0][J_IND];
  sJ[1] *= geof[1][J_IND];
  fscale[1] = sJ[1] / geof[1][J_IND];
}