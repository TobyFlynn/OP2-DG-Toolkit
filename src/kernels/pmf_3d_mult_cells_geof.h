inline void pmf_3d_mult_cells_geof(const int * __restrict__ p, const DG_FP * __restrict__ geof, DG_FP * __restrict__ in_x,
                                   DG_FP * __restrict__ in_y, DG_FP * __restrict__ in_z) {
  const int dg_np = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];

  const DG_FP _rx = geof[RX_IND];
  const DG_FP _ry = geof[RY_IND];
  const DG_FP _rz = geof[RZ_IND];
  const DG_FP _sx = geof[SX_IND];
  const DG_FP _sy = geof[SY_IND];
  const DG_FP _sz = geof[SZ_IND];
  const DG_FP _tx = geof[TX_IND];
  const DG_FP _ty = geof[TY_IND];
  const DG_FP _tz = geof[TZ_IND];
  for(int n = 0; n < dg_np; n++) {
    const DG_FP x = in_x[n];
    const DG_FP y = in_y[n];
    const DG_FP z = in_z[n];
    in_x[n] = _rx * x + _ry * y + _rz * z;
    in_y[n] = _sx * x + _sy * y + _sz * z;
    in_z[n] = _tx * x + _ty * y + _tz * z;
  }
}
