inline void poisson_coarse_matrix_3d_bop(const int *faceNum, const int *bc_type,
                                         const DG_FP *nx, const DG_FP *ny,
                                         const DG_FP *nz, const DG_FP *fscale,
                                         const DG_FP *sJ, const DG_FP *geof,
                                         DG_FP *op1) {
  // Do nothing for Neumann boundary conditions
  if(*bc_type == 1)
    return;

  // Handle Dirichlet boundary conditions
  const DG_FP *dr_mat = dg_Dr_kernel;
  const DG_FP *ds_mat = dg_Ds_kernel;
  const DG_FP *dt_mat = dg_Dt_kernel;
  const DG_FP *mmF0_mat = dg_MM_F0_kernel;
  const DG_FP *mmF1_mat = dg_MM_F1_kernel;
  const DG_FP *mmF2_mat = dg_MM_F2_kernel;
  const DG_FP *mmF3_mat = dg_MM_F3_kernel;

  const DG_FP *mmF;
  if(*faceNum == 0)
    mmF = mmF0_mat;
  else if(*faceNum == 1)
    mmF = mmF1_mat;
  else if(*faceNum == 2)
    mmF = mmF2_mat;
  else
    mmF = mmF3_mat;

  const int find = *faceNum * DG_NPF_N1;
  const int *fmask  = FMASK_TK;
  const int *fmaskB = &fmask[*faceNum * DG_NPF_N1];

  DG_FP D[DG_NP_N1 * DG_NP_N1];
  const DG_FP r_fact = nx[0] * geof[RX_IND] + ny[0] * geof[RY_IND] + nz[0] * geof[RZ_IND];
  const DG_FP s_fact = nx[0] * geof[SX_IND] + ny[0] * geof[SY_IND] + nz[0] * geof[SZ_IND];
  const DG_FP t_fact = nx[0] * geof[TX_IND] + ny[0] * geof[TY_IND] + nz[0] * geof[TZ_IND];
  for(int i = 0; i < DG_NP_N1; i++) {
    for(int j = 0; j < DG_NP_N1; j++) {
      // int ind = i + j * DG_NP_N1;
      int ind = DG_MAT_IND(i, j, DG_NP_N1, DG_NP_N1);
      D[ind] = r_fact * dr_mat[ind] + s_fact * ds_mat[ind] + t_fact * dt_mat[ind];
    }
  }

  const DG_FP gtau = 2.0 * (1 + 1) * (1 + 2) * *fscale;

  for(int i = 0; i < DG_NP_N1; i++) {
    for(int j = 0; j < DG_NP_N1; j++) {
      // int op_ind = i + j * DG_NP_N1;
      int op_ind = DG_MAT_IND(i, j, DG_NP_N1, DG_NP_N1);
      DG_FP tmp = 0.0;
      for(int k = 0; k < DG_NP_N1; k++) {
        // int a_ind0 = i + k * DG_NP_N1;
        int a_ind0 = DG_MAT_IND(i, k, DG_NP_N1, DG_NP_N1);
        // int a_ind1 = i * DG_NP_N1 + k;
        int a_ind1 = DG_MAT_IND(k, i, DG_NP_N1, DG_NP_N1);
        // int b_ind  = j * DG_NP_N1 + k;
        int b_ind = DG_MAT_IND(k, j, DG_NP_N1, DG_NP_N1);
        tmp += -*sJ * mmF[a_ind0] * D[b_ind] - D[a_ind1] * *sJ * mmF[b_ind];
      }
      op1[op_ind] += gtau * *sJ * mmF[op_ind] + tmp;
    }
  }
}
