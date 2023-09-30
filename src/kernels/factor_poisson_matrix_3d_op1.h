inline void factor_poisson_matrix_3d_op1(const int *order, const DG_FP *geof,
                                         const DG_FP *factor, DG_FP *op1) {
  const DG_FP *dr_mat = &dg_Dr_kernel[(*order - 1) * DG_NP * DG_NP];
  const DG_FP *ds_mat = &dg_Ds_kernel[(*order - 1) * DG_NP * DG_NP];
  const DG_FP *dt_mat = &dg_Dt_kernel[(*order - 1) * DG_NP * DG_NP];
  const DG_FP *mass_mat = &dg_Mass_kernel[(*order - 1) * DG_NP * DG_NP];
  const int dg_np = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS];

  DG_FP D[DG_NP * DG_NP], D_f[DG_NP * DG_NP], D_t[DG_NP * DG_NP];
  for(int i = 0; i < dg_np; i++) {
    for(int j = 0; j < dg_np; j++) {
      // int ind = i + j * dg_np;
      int ind = DG_MAT_IND(i, j, dg_np, dg_np);
      D[ind] = geof[RX_IND] * dr_mat[ind] + geof[SX_IND] * ds_mat[ind] + geof[TX_IND] * dt_mat[ind];
      D_f[ind] = D[ind] * factor[i];
    }
  }

  op2_in_kernel_gemm(false, false, dg_np, dg_np, dg_np, geof[J_IND], mass_mat, dg_np, D_f, dg_np, 0.0, D_t, dg_np);
  op2_in_kernel_gemm(false, false, dg_np, dg_np, dg_np, 1.0, D, dg_np, D_t, dg_np, 0.0, op1, dg_np);

  for(int i = 0; i < dg_np; i++) {
    for(int j = 0; j < dg_np; j++) {
      // int ind = i + j * dg_np;
      int ind = DG_MAT_IND(i, j, dg_np, dg_np);
      D[ind] = geof[RY_IND] * dr_mat[ind] + geof[SY_IND] * ds_mat[ind] + geof[TY_IND] * dt_mat[ind];
      D_f[ind] = D[ind] * factor[i];
    }
  }

  op2_in_kernel_gemm(false, false, dg_np, dg_np, dg_np, geof[J_IND], mass_mat, dg_np, D_f, dg_np, 0.0, D_t, dg_np);
  op2_in_kernel_gemm(false, false, dg_np, dg_np, dg_np, 1.0, D, dg_np, D_t, dg_np, 1.0, op1, dg_np);

  for(int i = 0; i < dg_np; i++) {
    for(int j = 0; j < dg_np; j++) {
      // int ind = i + j * dg_np;
      int ind = DG_MAT_IND(i, j, dg_np, dg_np);
      D[ind] = geof[RZ_IND] * dr_mat[ind] + geof[SZ_IND] * ds_mat[ind] + geof[TZ_IND] * dt_mat[ind];
      D_f[ind] = D[ind] * factor[i];
    }
  }

  op2_in_kernel_gemm(false, false, dg_np, dg_np, dg_np, geof[J_IND], mass_mat, dg_np, D_f, dg_np, 0.0, D_t, dg_np);
  op2_in_kernel_gemm(false, false, dg_np, dg_np, dg_np, 1.0, D, dg_np, D_t, dg_np, 1.0, op1, dg_np);
}
