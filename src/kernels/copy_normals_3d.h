inline void copy_normals_3d(const int *faceNum, const DG_FP *nx,
                            const DG_FP *ny, const DG_FP *nz, const DG_FP *sJ,
                            DG_FP **nx_c, DG_FP **ny_c, DG_FP **nz_c,
                            DG_FP **sJ_c) {
  const int faceNumL = faceNum[0];
  nx_c[0][faceNumL] = nx[0];
  ny_c[0][faceNumL] = ny[0];
  nz_c[0][faceNumL] = nz[0];
  sJ_c[0][faceNumL] = sJ[0];

  const int faceNumR = faceNum[1];
  nx_c[1][faceNumR] = nx[1];
  ny_c[1][faceNumR] = ny[1];
  nz_c[1][faceNumR] = nz[1];
  sJ_c[1][faceNumR] = sJ[1];
}
