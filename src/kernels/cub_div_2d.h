inline void cub_div_2d(const int *p, const DG_FP *rx, const DG_FP *sx,
                       const DG_FP *ry, const DG_FP *sy, const DG_FP *J,
                       DG_FP *temp0, const DG_FP *temp1, const DG_FP *temp2,
                       const DG_FP *temp3) {
  // Get constants for this element's order
  const int dg_cub_np = DG_CONSTANTS_TK[(*p - 1) * 5 + 2];
  const DG_FP *cubW  = &cubW_g_TK[(*p - 1) * DG_CUB_NP];

  for(int i = 0; i < dg_cub_np; i++) {
    DG_FP div = rx[i] * temp0[i] + sx[i] * temp1[i];
    div += ry[i] * temp2[i] + sy[i] * temp3[i];
    temp0[i] = cubW[i] * J[i] * div;
  }
}
