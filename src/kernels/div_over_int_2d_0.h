inline void div_over_int_2d_0(const DG_FP *geof, DG_FP *u, DG_FP *v) {
  for(int i = 0; i < DG_CUB_2D_NP; i++) {
    const DG_FP _u = u[i]; const DG_FP _v = v[i];
    u[i] = geof[RX_IND] * _u + geof[RY_IND] * _v;
    v[i] = geof[SX_IND] * _u + geof[SY_IND] * _v;
  }
}