inline void copy_normals_2d(const int **order, const int *faceNum,
                            const DG_FP **nx_c, const DG_FP **ny_c,
                            const DG_FP **sJ_c, const DG_FP **fscale_c,
                            DG_FP *nx, DG_FP *ny, DG_FP *sJ, DG_FP *fscale) {
  const int p = order[0][0];
  const int dg_npf = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS + 1];
  const int faceNumL = faceNum[0];
  const int faceNumR = faceNum[1];
  nx[0] = nx_c[0][faceNumL * dg_npf];
  ny[0] = ny_c[0][faceNumL * dg_npf];
  sJ[0] = sJ_c[0][faceNumL * dg_npf];
  fscale[0] = fscale_c[0][faceNumL * dg_npf];
  nx[1] = nx_c[1][faceNumR * dg_npf];
  ny[1] = ny_c[1][faceNumR * dg_npf];
  sJ[1] = sJ_c[1][faceNumR * dg_npf];
  fscale[1] = fscale_c[1][faceNumR * dg_npf];
}
