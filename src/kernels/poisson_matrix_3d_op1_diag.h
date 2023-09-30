inline void poisson_matrix_3d_op1_diag(const int *order, const DG_FP *geof,
                                       DG_FP *diag) {
  const DG_FP *dr_mat = &dg_Dr_kernel[(*order - 1) * DG_NP * DG_NP];
  const DG_FP *ds_mat = &dg_Ds_kernel[(*order - 1) * DG_NP * DG_NP];
  const DG_FP *dt_mat = &dg_Dt_kernel[(*order - 1) * DG_NP * DG_NP];
  const DG_FP *mass_mat = &dg_Mass_kernel[(*order - 1) * DG_NP * DG_NP];
  const int dg_np = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS];

  DG_FP Dx[DG_NP * DG_NP], Dy[DG_NP * DG_NP], Dz[DG_NP * DG_NP];
  for(int i = 0; i < dg_np; i++) {
    for(int j = 0; j < dg_np; j++) {
      // int ind = i + j * dg_np;
      int ind = DG_MAT_IND(i, j, dg_np, dg_np);
      Dx[ind] = geof[RX_IND] * dr_mat[ind] + geof[SX_IND] * ds_mat[ind] + geof[TX_IND] * dt_mat[ind];
      Dy[ind] = geof[RY_IND] * dr_mat[ind] + geof[SY_IND] * ds_mat[ind] + geof[TY_IND] * dt_mat[ind];
      Dz[ind] = geof[RZ_IND] * dr_mat[ind] + geof[SZ_IND] * ds_mat[ind] + geof[TZ_IND] * dt_mat[ind];
    }
  }

  DG_FP Dx_t[DG_NP * DG_NP], Dy_t[DG_NP * DG_NP], Dz_t[DG_NP * DG_NP];
  op2_in_kernel_gemm(false, false, dg_np, dg_np, dg_np, geof[J_IND], mass_mat, dg_np, Dx, dg_np, 0.0, Dx_t, dg_np);
  op2_in_kernel_gemm(false, false, dg_np, dg_np, dg_np, geof[J_IND], mass_mat, dg_np, Dy, dg_np, 0.0, Dy_t, dg_np);
  op2_in_kernel_gemm(false, false, dg_np, dg_np, dg_np, geof[J_IND], mass_mat, dg_np, Dz, dg_np, 0.0, Dz_t, dg_np);

  for(int i = 0; i < dg_np; i++) {
    diag[i] = 0.0;
    for(int k = 0; k < dg_np; k++) {
      // int a_ind = i * dg_np + k;
      int a_ind = DG_MAT_IND(k, i, dg_np, dg_np);
      // int b_ind = i * dg_np + k;
      int b_ind = DG_MAT_IND(k, i, dg_np, dg_np);
      diag[i] += Dx[a_ind] * Dx_t[b_ind] + Dy[a_ind] * Dy_t[b_ind] + Dz[a_ind] * Dz_t[b_ind];
    }
  }
}
