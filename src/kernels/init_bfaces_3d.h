inline void init_bfaces_3d(const int *faceNum, const DG_FP *rx, const DG_FP *ry,
                           const DG_FP *rz, const DG_FP *sx, const DG_FP *sy,
                           const DG_FP *sz, const DG_FP *tx, const DG_FP *ty,
                           const DG_FP *tz, const DG_FP *J, DG_FP *nx,
                           DG_FP *ny, DG_FP *nz, DG_FP *sJ, DG_FP *fscale) {
  if(*faceNum == 0) {
    *nx = -*tx;
    *ny = -*ty;
    *nz = -*tz;
  } else if(*faceNum == 1) {
    *nx = -*sx;
    *ny = -*sy;
    *nz = -*sz;
  } else if(*faceNum == 2) {
    *nx = *rx + *sx + *tx;
    *ny = *ry + *sy + *ty;
    *nz = *rz + *sz + *tz;
  } else {
    *nx = -*rx;
    *ny = -*ry;
    *nz = -*rz;
  }

  *sJ = sqrt(*nx * *nx + *ny * *ny + *nz * *nz);
  *nx /= *sJ;
  *ny /= *sJ;
  *nz /= *sJ;
  *sJ *= *J;
  *fscale = *sJ / *J;
}