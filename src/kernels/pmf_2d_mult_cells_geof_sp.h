inline void pmf_2d_mult_cells_geof_sp(const int *p, const DG_FP *geof, float *in_x,
                                      float *in_y) {
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  const float rx = (float)geof[RX_IND];
  const float sx = (float)geof[SX_IND];
  const float ry = (float)geof[RY_IND];
  const float sy = (float)geof[SY_IND];
  for(int n = 0; n < dg_np; n++) {
    const float x = in_x[n];
    const float y = in_y[n];
    in_x[n] = rx * x + ry * y;
    in_y[n] = sx * x + sy * y;
  }
}
