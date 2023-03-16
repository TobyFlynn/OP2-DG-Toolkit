inline void pre_init_gauss(const int *p, const DG_FP *x, const DG_FP *y,
                           const DG_FP *gF0Dr_g, const DG_FP *gF0Ds_g,
                           const DG_FP *gF1Dr_g, const DG_FP *gF1Ds_g,
                           const DG_FP *gF2Dr_g, const DG_FP *gF2Ds_g,
                           DG_FP *rx, DG_FP *sx, DG_FP *ry, DG_FP *sy) {
  const int dg_np    = DG_CONSTANTS_TK[(*p - 1) * 5];

  const DG_FP *gF0Dr = &gF0Dr_g[(*p - 1) * DG_GF_NP * DG_NP];
  const DG_FP *gF0Ds = &gF0Ds_g[(*p - 1) * DG_GF_NP * DG_NP];
  const DG_FP *gF1Dr = &gF1Dr_g[(*p - 1) * DG_GF_NP * DG_NP];
  const DG_FP *gF1Ds = &gF1Ds_g[(*p - 1) * DG_GF_NP * DG_NP];
  const DG_FP *gF2Dr = &gF2Dr_g[(*p - 1) * DG_GF_NP * DG_NP];
  const DG_FP *gF2Ds = &gF2Ds_g[(*p - 1) * DG_GF_NP * DG_NP];

  op2_in_kernel_gemv(false, DG_GF_NP, dg_np, 1.0, gF0Dr, DG_GF_NP, x, 0.0, rx);
  op2_in_kernel_gemv(false, DG_GF_NP, dg_np, 1.0, gF0Ds, DG_GF_NP, x, 0.0, sx);
  op2_in_kernel_gemv(false, DG_GF_NP, dg_np, 1.0, gF0Dr, DG_GF_NP, y, 0.0, ry);
  op2_in_kernel_gemv(false, DG_GF_NP, dg_np, 1.0, gF0Ds, DG_GF_NP, y, 0.0, sy);

  op2_in_kernel_gemv(false, DG_GF_NP, dg_np, 1.0, gF1Dr, DG_GF_NP, x, 0.0, &rx[DG_GF_NP]);
  op2_in_kernel_gemv(false, DG_GF_NP, dg_np, 1.0, gF1Ds, DG_GF_NP, x, 0.0, &sx[DG_GF_NP]);
  op2_in_kernel_gemv(false, DG_GF_NP, dg_np, 1.0, gF1Dr, DG_GF_NP, y, 0.0, &ry[DG_GF_NP]);
  op2_in_kernel_gemv(false, DG_GF_NP, dg_np, 1.0, gF1Ds, DG_GF_NP, y, 0.0, &sy[DG_GF_NP]);

  op2_in_kernel_gemv(false, DG_GF_NP, dg_np, 1.0, gF2Dr, DG_GF_NP, x, 0.0, &rx[2 * DG_GF_NP]);
  op2_in_kernel_gemv(false, DG_GF_NP, dg_np, 1.0, gF2Ds, DG_GF_NP, x, 0.0, &sx[2 * DG_GF_NP]);
  op2_in_kernel_gemv(false, DG_GF_NP, dg_np, 1.0, gF2Dr, DG_GF_NP, y, 0.0, &ry[2 * DG_GF_NP]);
  op2_in_kernel_gemv(false, DG_GF_NP, dg_np, 1.0, gF2Ds, DG_GF_NP, y, 0.0, &sy[2 * DG_GF_NP]);
}
