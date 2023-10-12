inline void copy_normals_bface_2d(const int *faceNum, const DG_FP *nx_c, const DG_FP *ny_c,
                                  const DG_FP *sJ_c, const DG_FP *fscale_c, DG_FP *nx, 
                                  DG_FP *ny, DG_FP *sJ, DG_FP *fscale) {
  const int _faceNum = faceNum[0];
  nx[0] = nx_c[_faceNum];
  ny[0] = ny_c[_faceNum];
  sJ[0] = sJ_c[_faceNum];
  fscale[0] = fscale_c[_faceNum];
}
