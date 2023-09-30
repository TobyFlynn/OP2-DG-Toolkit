inline void poisson_coarse_matrix_3d_op1(const DG_FP *geof, DG_FP *op1) {
  const DG_FP *dr_mat = dg_Dr_kernel;
  const DG_FP *ds_mat = dg_Ds_kernel;
  const DG_FP *dt_mat = dg_Dt_kernel;
  const DG_FP *mass_mat = dg_Mass_kernel;

  DG_FP Dx[DG_NP_N1 * DG_NP_N1], Dy[DG_NP_N1 * DG_NP_N1], Dz[DG_NP_N1 * DG_NP_N1];
  for(int i = 0; i < DG_NP_N1; i++) {
    for(int j = 0; j < DG_NP_N1; j++) {
      // int ind = i + j * dg_np;
      int ind = DG_MAT_IND(i, j, DG_NP_N1, DG_NP_N1);
      Dx[ind] = geof[RX_IND] * dr_mat[ind] + geof[SX_IND] * ds_mat[ind] + geof[TX_IND] * dt_mat[ind];
      Dy[ind] = geof[RY_IND] * dr_mat[ind] + geof[SY_IND] * ds_mat[ind] + geof[TY_IND] * dt_mat[ind];
      Dz[ind] = geof[RZ_IND] * dr_mat[ind] + geof[SZ_IND] * ds_mat[ind] + geof[TZ_IND] * dt_mat[ind];
    }
  }

  DG_FP Dx_t[DG_NP_N1 * DG_NP_N1], Dy_t[DG_NP_N1 * DG_NP_N1], Dz_t[DG_NP_N1 * DG_NP_N1];
  op2_in_kernel_gemm(false, false, DG_NP_N1, DG_NP_N1, DG_NP_N1, 1.0, mass_mat, DG_NP_N1, Dx, DG_NP_N1, 0.0, Dx_t, DG_NP_N1);
  op2_in_kernel_gemm(false, false, DG_NP_N1, DG_NP_N1, DG_NP_N1, 1.0, mass_mat, DG_NP_N1, Dy, DG_NP_N1, 0.0, Dy_t, DG_NP_N1);
  op2_in_kernel_gemm(false, false, DG_NP_N1, DG_NP_N1, DG_NP_N1, 1.0, mass_mat, DG_NP_N1, Dz, DG_NP_N1, 0.0, Dz_t, DG_NP_N1);

  op2_in_kernel_gemm(true, false, DG_NP_N1, DG_NP_N1, DG_NP_N1, geof[J_IND], Dx, DG_NP_N1, Dx_t, DG_NP_N1, 0.0, op1, DG_NP_N1);
  op2_in_kernel_gemm(true, false, DG_NP_N1, DG_NP_N1, DG_NP_N1, geof[J_IND], Dy, DG_NP_N1, Dy_t, DG_NP_N1, 1.0, op1, DG_NP_N1);
  op2_in_kernel_gemm(true, false, DG_NP_N1, DG_NP_N1, DG_NP_N1, geof[J_IND], Dz, DG_NP_N1, Dz_t, DG_NP_N1, 1.0, op1, DG_NP_N1);
}
