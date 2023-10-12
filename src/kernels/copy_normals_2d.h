inline void copy_normals_2d(const int *faceNum, const DG_FP **nx_c, const DG_FP **ny_c,
                            const DG_FP **sJ_c, const DG_FP **fscale_c, DG_FP *nx, 
                            DG_FP *ny, DG_FP *sJ, DG_FP *fscale) {
  const int faceNumL = faceNum[0];
  const int faceNumR = faceNum[1];
  nx[0] = nx_c[0][faceNumL];
  ny[0] = ny_c[0][faceNumL];
  sJ[0] = sJ_c[0][faceNumL];
  fscale[0] = fscale_c[0][faceNumL];
  nx[1] = nx_c[1][faceNumR];
  ny[1] = ny_c[1][faceNumR];
  sJ[1] = sJ_c[1][faceNumR];
  fscale[1] = fscale_c[1][faceNumR];
}
