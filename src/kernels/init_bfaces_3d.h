inline void init_bfaces_3d(const int *faceNum, const DG_FP *geof, DG_FP *nx,
                           DG_FP *ny, DG_FP *nz, DG_FP *sJ, DG_FP *fscale) {
  if(*faceNum == 0) {
    *nx = -geof[TX_IND];
    *ny = -geof[TY_IND];
    *nz = -geof[TZ_IND];
  } else if(*faceNum == 1) {
    *nx = -geof[SX_IND];
    *ny = -geof[SY_IND];
    *nz = -geof[SZ_IND];
  } else if(*faceNum == 2) {
    *nx = geof[RX_IND] + geof[SX_IND] + geof[TX_IND];
    *ny = geof[RY_IND] + geof[SY_IND] + geof[TY_IND];
    *nz = geof[RZ_IND] + geof[SZ_IND] + geof[TZ_IND];
  } else {
    *nx = -geof[RX_IND];
    *ny = -geof[RY_IND];
    *nz = -geof[RZ_IND];
  }

  *sJ = sqrt(*nx * *nx + *ny * *ny + *nz * *nz);
  *nx /= *sJ;
  *ny /= *sJ;
  *nz /= *sJ;
  *sJ *= geof[J_IND];
  *fscale = *sJ / geof[J_IND];
}