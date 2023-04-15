inline void normals_check_3d(const int **order, const int *faceNum, DG_FP *nx,
                          DG_FP *ny, DG_FP *nz, const DG_FP **x,
                          const DG_FP **y, const DG_FP **z, const DG_FP **nodeX_,
                          const DG_FP **nodeY_, const DG_FP **nodeZ_, int *errors) {
  const int dg_npf = DG_CONSTANTS_TK[(order[0][0] - 1) * 2 + 1];
  const int *fmask  = &FMASK_TK[(order[0][0] - 1) * 4 * DG_NPF];
  const int *fmaskL = &fmask[faceNum[0] * dg_npf];
  const int *fmaskR = &fmask[faceNum[1] * dg_npf];

  // Left face
  DG_FP av_x, av_y, av_z;
  av_x = 0.0; av_y = 0.0; av_z = 0.0;
  for(int i = 0; i < dg_npf; i++) {
    const int fmaskL_ind = fmaskL[i];
    av_x += x[0][fmaskL_ind];
    av_y += y[0][fmaskL_ind];
    av_z += z[0][fmaskL_ind];
  }
  av_x /= (DG_FP)dg_npf;
  av_y /= (DG_FP)dg_npf;
  av_z /= (DG_FP)dg_npf;

  DG_FP ptX = av_x + 1e-5 * nx[0];
  DG_FP ptY = av_y + 1e-5 * ny[0];
  DG_FP ptZ = av_z + 1e-5 * nz[0];

  bool sameSide0, sameSide1, sameSide2, sameSide3;
  DG_FP normal[3];
  // (v1 - v0) x (v2 - v0)
  normal[0] = (nodeY_[0][1] - nodeY_[0][0]) * (nodeZ_[0][2] - nodeZ_[0][0]) - (nodeZ_[0][1] - nodeZ_[0][0]) * (nodeY_[0][2] - nodeY_[0][0]);
  normal[1] = (nodeZ_[0][1] - nodeZ_[0][0]) * (nodeX_[0][2] - nodeX_[0][0]) - (nodeX_[0][1] - nodeX_[0][0]) * (nodeZ_[0][2] - nodeZ_[0][0]);
  normal[2] = (nodeX_[0][1] - nodeX_[0][0]) * (nodeY_[0][2] - nodeY_[0][0]) - (nodeY_[0][1] - nodeY_[0][0]) * (nodeX_[0][2] - nodeX_[0][0]);
  // normal . (v3 - v0)
  DG_FP dotV = normal[0] * (nodeX_[0][3] - nodeX_[0][0]) + normal[1] * (nodeY_[0][3] - nodeY_[0][0]) + normal[2] * (nodeZ_[0][3] - nodeZ_[0][0]);
  // normal . (p - v0)
  DG_FP dotP = normal[0] * (ptX - nodeX_[0][0]) + normal[1] * (ptY - nodeY_[0][0]) + normal[2] * (ptZ - nodeZ_[0][0]);
  sameSide0 = (dotV > 0.0) == (dotP > 0.0);

  // (v2 - v1) x (v3 - v1)
  normal[0] = (nodeY_[0][2] - nodeY_[0][1]) * (nodeZ_[0][3] - nodeZ_[0][1]) - (nodeZ_[0][2] - nodeZ_[0][1]) * (nodeY_[0][3] - nodeY_[0][1]);
  normal[1] = (nodeZ_[0][2] - nodeZ_[0][1]) * (nodeX_[0][3] - nodeX_[0][1]) - (nodeX_[0][2] - nodeX_[0][1]) * (nodeZ_[0][3] - nodeZ_[0][1]);
  normal[2] = (nodeX_[0][2] - nodeX_[0][1]) * (nodeY_[0][3] - nodeY_[0][1]) - (nodeY_[0][2] - nodeY_[0][1]) * (nodeX_[0][3] - nodeX_[0][1]);
  // normal . (v0 - v1)
  dotV = normal[0] * (nodeX_[0][0] - nodeX_[0][1]) + normal[1] * (nodeY_[0][0] - nodeY_[0][1]) + normal[2] * (nodeZ_[0][0] - nodeZ_[0][1]);
  // normal . (p - v1)
  dotP = normal[0] * (ptX - nodeX_[0][1]) + normal[1] * (ptY - nodeY_[0][1]) + normal[2] * (ptZ - nodeZ_[0][1]);
  sameSide1 = (dotV > 0.0) == (dotP > 0.0);

  // (v3 - v2) x (v0 - v2)
  normal[0] = (nodeY_[0][3] - nodeY_[0][2]) * (nodeZ_[0][0] - nodeZ_[0][2]) - (nodeZ_[0][3] - nodeZ_[0][2]) * (nodeY_[0][0] - nodeY_[0][2]);
  normal[1] = (nodeZ_[0][3] - nodeZ_[0][2]) * (nodeX_[0][0] - nodeX_[0][2]) - (nodeX_[0][3] - nodeX_[0][2]) * (nodeZ_[0][0] - nodeZ_[0][2]);
  normal[2] = (nodeX_[0][3] - nodeX_[0][2]) * (nodeY_[0][0] - nodeY_[0][2]) - (nodeY_[0][3] - nodeY_[0][2]) * (nodeX_[0][0] - nodeX_[0][2]);
  // normal . (v1 - v2)
  dotV = normal[0] * (nodeX_[0][1] - nodeX_[0][2]) + normal[1] * (nodeY_[0][1] - nodeY_[0][2]) + normal[2] * (nodeZ_[0][1] - nodeZ_[0][2]);
  // normal . (p - v2)
  dotP = normal[0] * (ptX - nodeX_[0][2]) + normal[1] * (ptY - nodeY_[0][2]) + normal[2] * (ptZ - nodeZ_[0][2]);
  sameSide2 = (dotV > 0.0) == (dotP > 0.0);

  // (v0 - v3) x (v1 - v3)
  normal[0] = (nodeY_[0][0] - nodeY_[0][3]) * (nodeZ_[0][1] - nodeZ_[0][3]) - (nodeZ_[0][0] - nodeZ_[0][3]) * (nodeY_[0][1] - nodeY_[0][3]);
  normal[1] = (nodeZ_[0][0] - nodeZ_[0][3]) * (nodeX_[0][1] - nodeX_[0][3]) - (nodeX_[0][0] - nodeX_[0][3]) * (nodeZ_[0][1] - nodeZ_[0][3]);
  normal[2] = (nodeX_[0][0] - nodeX_[0][3]) * (nodeY_[0][1] - nodeY_[0][3]) - (nodeY_[0][0] - nodeY_[0][3]) * (nodeX_[0][1] - nodeX_[0][3]);
  // normal . (v2 - v3)
  dotV = normal[0] * (nodeX_[0][2] - nodeX_[0][3]) + normal[1] * (nodeY_[0][2] - nodeY_[0][3]) + normal[2] * (nodeZ_[0][2] - nodeZ_[0][3]);
  // normal . (p - v3)
  dotP = normal[0] * (ptX - nodeX_[0][3]) + normal[1] * (ptY - nodeY_[0][3]) + normal[2] * (ptZ - nodeZ_[0][3]);
  sameSide3 = (dotV > 0.0) == (dotP > 0.0);

  if(sameSide0 && sameSide1 && sameSide2 && sameSide3) {
    *errors += 1;
    // nx[0] = -nx[0];
  }

  // Right face
  av_x = 0.0; av_y = 0.0; av_z = 0.0;
  for(int i = 0; i < dg_npf; i++) {
    const int fmaskR_ind = fmaskR[i];
    av_x += x[1][fmaskR_ind];
    av_y += y[1][fmaskR_ind];
    av_z += z[1][fmaskR_ind];
  }
  av_x /= (DG_FP)dg_npf;
  av_y /= (DG_FP)dg_npf;
  av_z /= (DG_FP)dg_npf;

  ptX = av_x + 1e-5 * nx[1];
  ptY = av_y + 1e-5 * ny[1];
  ptZ = av_z + 1e-5 * nz[1];

  // (v1 - v0) x (v2 - v0)
  normal[0] = (nodeY_[1][1] - nodeY_[1][0]) * (nodeZ_[1][2] - nodeZ_[1][0]) - (nodeZ_[1][1] - nodeZ_[1][0]) * (nodeY_[1][2] - nodeY_[1][0]);
  normal[1] = (nodeZ_[1][1] - nodeZ_[1][0]) * (nodeX_[1][2] - nodeX_[1][0]) - (nodeX_[1][1] - nodeX_[1][0]) * (nodeZ_[1][2] - nodeZ_[1][0]);
  normal[2] = (nodeX_[1][1] - nodeX_[1][0]) * (nodeY_[1][2] - nodeY_[1][0]) - (nodeY_[1][1] - nodeY_[1][0]) * (nodeX_[1][2] - nodeX_[1][0]);
  // normal . (v3 - v0)
  dotV = normal[0] * (nodeX_[1][3] - nodeX_[1][0]) + normal[1] * (nodeY_[1][3] - nodeY_[1][0]) + normal[2] * (nodeZ_[1][3] - nodeZ_[1][0]);
  // normal . (p - v0)
  dotP = normal[0] * (ptX - nodeX_[1][0]) + normal[1] * (ptY - nodeY_[1][0]) + normal[2] * (ptZ - nodeZ_[1][0]);
  sameSide0 = (dotV > 0.0) == (dotP > 0.0);

  // (v2 - v1) x (v3 - v1)
  normal[0] = (nodeY_[1][2] - nodeY_[1][1]) * (nodeZ_[1][3] - nodeZ_[1][1]) - (nodeZ_[1][2] - nodeZ_[1][1]) * (nodeY_[1][3] - nodeY_[1][1]);
  normal[1] = (nodeZ_[1][2] - nodeZ_[1][1]) * (nodeX_[1][3] - nodeX_[1][1]) - (nodeX_[1][2] - nodeX_[1][1]) * (nodeZ_[1][3] - nodeZ_[1][1]);
  normal[2] = (nodeX_[1][2] - nodeX_[1][1]) * (nodeY_[1][3] - nodeY_[1][1]) - (nodeY_[1][2] - nodeY_[1][1]) * (nodeX_[1][3] - nodeX_[1][1]);
  // normal . (v0 - v1)
  dotV = normal[0] * (nodeX_[1][0] - nodeX_[1][1]) + normal[1] * (nodeY_[1][0] - nodeY_[1][1]) + normal[2] * (nodeZ_[1][0] - nodeZ_[1][1]);
  // normal . (p - v1)
  dotP = normal[0] * (ptX - nodeX_[1][1]) + normal[1] * (ptY - nodeY_[1][1]) + normal[2] * (ptZ - nodeZ_[1][1]);
  sameSide1 = (dotV > 0.0) == (dotP > 0.0);

  // (v3 - v2) x (v0 - v2)
  normal[0] = (nodeY_[1][3] - nodeY_[1][2]) * (nodeZ_[1][0] - nodeZ_[1][2]) - (nodeZ_[1][3] - nodeZ_[1][2]) * (nodeY_[1][0] - nodeY_[1][2]);
  normal[1] = (nodeZ_[1][3] - nodeZ_[1][2]) * (nodeX_[1][0] - nodeX_[1][2]) - (nodeX_[1][3] - nodeX_[1][2]) * (nodeZ_[1][0] - nodeZ_[1][2]);
  normal[2] = (nodeX_[1][3] - nodeX_[1][2]) * (nodeY_[1][0] - nodeY_[1][2]) - (nodeY_[1][3] - nodeY_[1][2]) * (nodeX_[1][0] - nodeX_[1][2]);
  // normal . (v1 - v2)
  dotV = normal[0] * (nodeX_[1][1] - nodeX_[1][2]) + normal[1] * (nodeY_[1][1] - nodeY_[1][2]) + normal[2] * (nodeZ_[1][1] - nodeZ_[1][2]);
  // normal . (p - v2)
  dotP = normal[0] * (ptX - nodeX_[1][2]) + normal[1] * (ptY - nodeY_[1][2]) + normal[2] * (ptZ - nodeZ_[1][2]);
  sameSide2 = (dotV > 0.0) == (dotP > 0.0);

  // (v0 - v3) x (v1 - v3)
  normal[0] = (nodeY_[1][0] - nodeY_[1][3]) * (nodeZ_[1][1] - nodeZ_[1][3]) - (nodeZ_[1][0] - nodeZ_[1][3]) * (nodeY_[1][1] - nodeY_[1][3]);
  normal[1] = (nodeZ_[1][0] - nodeZ_[1][3]) * (nodeX_[1][1] - nodeX_[1][3]) - (nodeX_[1][0] - nodeX_[1][3]) * (nodeZ_[1][1] - nodeZ_[1][3]);
  normal[2] = (nodeX_[1][0] - nodeX_[1][3]) * (nodeY_[1][1] - nodeY_[1][3]) - (nodeY_[1][0] - nodeY_[1][3]) * (nodeX_[1][1] - nodeX_[1][3]);
  // normal . (v2 - v3)
  dotV = normal[0] * (nodeX_[1][2] - nodeX_[1][3]) + normal[1] * (nodeY_[1][2] - nodeY_[1][3]) + normal[2] * (nodeZ_[1][2] - nodeZ_[1][3]);
  // normal . (p - v3)
  dotP = normal[0] * (ptX - nodeX_[1][3]) + normal[1] * (ptY - nodeY_[1][3]) + normal[2] * (ptZ - nodeZ_[1][3]);
  sameSide3 = (dotV > 0.0) == (dotP > 0.0);

  if(sameSide0 && sameSide1 && sameSide2 && sameSide3) {
    *errors += 1;
    // nx[1] = -nx[1];
  }

  if(!fp_equal(nx[0], -nx[1]) || !fp_equal(ny[0], -ny[1]) || !fp_equal(nz[0], -nz[1]))
    *errors += 1;
}
