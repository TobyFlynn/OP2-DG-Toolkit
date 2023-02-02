inline void init_bfaces_3d(const int *faceNum, const double *rx, const double *ry,
                           const double *rz, const double *sx, const double *sy,
                           const double *sz, const double *tx, const double *ty,
                           const double *tz, const double *J, double *nx,
                           double *ny, double *nz, double *sJ, double *fscale) {
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