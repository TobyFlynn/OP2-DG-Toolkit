inline void copy_normals_bface_2d(const int *order, const int *faceNum,
                                  const DG_FP *nx_c, const DG_FP *ny_c,
                                  const DG_FP *sJ_c, const DG_FP *fscale_c,
                                  DG_FP *nx, DG_FP *ny, DG_FP *sJ,
                                  DG_FP *fscale) {
  const int p = order[0];
  const int dg_npf = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS + 1];
  const int _faceNum = faceNum[0];
  nx[0] = nx_c[_faceNum * dg_npf];
  ny[0] = ny_c[_faceNum * dg_npf];
  sJ[0] = sJ_c[_faceNum * dg_npf];
  fscale[0] = fscale_c[_faceNum * dg_npf];
}
