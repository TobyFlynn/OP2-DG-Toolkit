inline void cub_grad_weak_2d(const int *p, DG_FP *temp0, const DG_FP *rx,
                             const DG_FP *sx, const DG_FP *ry, const DG_FP *sy,
                             const DG_FP *J, DG_FP *temp1, DG_FP *temp2,
                             DG_FP *temp3) {
  // Get constants for this element's order
  const int dg_cub_np = DG_CONSTANTS[(*p - 1) * 5 + 2];
  const DG_FP *cubW  = &cubW_g[(*p - 1) * DG_CUB_NP];

  for(int i = 0; i < dg_cub_np; i++) {
    temp1[i] = cubW[i] * J[i] * sx[i] * temp0[i];
    temp2[i] = cubW[i] * J[i] * ry[i] * temp0[i];
    temp3[i] = cubW[i] * J[i] * sy[i] * temp0[i];
    temp0[i] = cubW[i] * J[i] * rx[i] * temp0[i];
  }
}
