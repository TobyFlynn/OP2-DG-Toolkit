inline void copy_geof_2d(const DG_FP *rx, const DG_FP *ry, const DG_FP *sx,
                         const DG_FP *sy, const DG_FP *J, DG_FP *geof) {
  geof[RX_IND] = rx[0];
  geof[RY_IND] = ry[0];
  geof[SX_IND] = sx[0];
  geof[SY_IND] = sy[0];
  geof[J_IND]  = J[0];
}
