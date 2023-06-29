inline void copy_normals_bfaces_3d(const int *faceNum, const DG_FP *nx,
                                   const DG_FP *ny, const DG_FP *nz, const DG_FP *sJ,
                                   DG_FP *nx_c, DG_FP *ny_c, DG_FP *nz_c,
                                   DG_FP *sJ_c) {
  const int _faceNum = faceNum[0];
  nx_c[_faceNum] = nx[0];
  ny_c[_faceNum] = ny[0];
  nz_c[_faceNum] = nz[0];
  sJ_c[_faceNum] = sJ[0];
}
