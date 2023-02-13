inline void init_faces_3d(const int *faceNum, const DG_FP **rx, const DG_FP **ry,
                          const DG_FP **rz, const DG_FP **sx, const DG_FP **sy,
                          const DG_FP **sz, const DG_FP **tx, const DG_FP **ty,
                          const DG_FP **tz, const DG_FP **J, DG_FP *nx,
                          DG_FP *ny, DG_FP *nz, DG_FP *sJ, DG_FP *fscale) {
  if(faceNum[0] == 0) {
    nx[0] = -tx[0][0];
    ny[0] = -ty[0][0];
    nz[0] = -tz[0][0];
  } else if(faceNum[0] == 1) {
    nx[0] = -sx[0][0];
    ny[0] = -sy[0][0];
    nz[0] = -sz[0][0];
  } else if(faceNum[0] == 2) {
    nx[0] = rx[0][0] + sx[0][0] + tx[0][0];
    ny[0] = ry[0][0] + sy[0][0] + ty[0][0];
    nz[0] = rz[0][0] + sz[0][0] + tz[0][0];
  } else {
    nx[0] = -rx[0][0];
    ny[0] = -ry[0][0];
    nz[0] = -rz[0][0];
  }

  if(faceNum[1] == 0) {
    nx[1] = -tx[1][0];
    ny[1] = -ty[1][0];
    nz[1] = -tz[1][0];
  } else if(faceNum[1] == 1) {
    nx[1] = -sx[1][0];
    ny[1] = -sy[1][0];
    nz[1] = -sz[1][0];
  } else if(faceNum[1] == 2) {
    nx[1] = rx[1][0] + sx[1][0] + tx[1][0];
    ny[1] = ry[1][0] + sy[1][0] + ty[1][0];
    nz[1] = rz[1][0] + sz[1][0] + tz[1][0];
  } else {
    nx[1] = -rx[1][0];
    ny[1] = -ry[1][0];
    nz[1] = -rz[1][0];
  }

  sJ[0] = sqrt(nx[0] * nx[0] + ny[0] * ny[0] + nz[0] * nz[0]);
  sJ[1] = sqrt(nx[1] * nx[1] + ny[1] * ny[1] + nz[1] * nz[1]);
  nx[0] /= sJ[0];
  ny[0] /= sJ[0];
  nz[0] /= sJ[0];
  nx[1] /= sJ[1];
  ny[1] /= sJ[1];
  nz[1] /= sJ[1];
  sJ[0] *= J[0][0];
  fscale[0] = sJ[0] / J[0][0];
  sJ[1] *= J[1][0];
  fscale[1] = sJ[1] / J[1][0];
}