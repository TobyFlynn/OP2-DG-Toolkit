inline void poisson_matrix_2d_op2_block_diag(const int *order, const int *faceNum,
                          const bool *reversed, const DG_FP *nx, const DG_FP *ny, 
                          const DG_FP *fscale, const DG_FP *sJ, const DG_FP **geof,
                          DG_FP *op1L, DG_FP *op1R) {
  const DG_FP *dr_mat = dg_Dr_kernel + (*order - 1) * DG_NP * DG_NP;
  const DG_FP *ds_mat = dg_Ds_kernel + (*order - 1) * DG_NP * DG_NP;

  const DG_FP *mmF0_mat = dg_MM_F0_kernel + (*order - 1) * DG_NP * DG_NP;
  const DG_FP *mmF1_mat = dg_MM_F1_kernel + (*order - 1) * DG_NP * DG_NP;
  const DG_FP *mmF2_mat = dg_MM_F2_kernel + (*order - 1) * DG_NP * DG_NP;

  const bool rev = *reversed;

  const DG_FP *mmFL, *mmFR;
  if(faceNum[0] == 0)
    mmFL = mmF0_mat;
  else if(faceNum[0] == 1)
    mmFL = mmF1_mat;
  else
    mmFL = mmF2_mat;

  if(faceNum[1] == 0)
    mmFR = mmF0_mat;
  else if(faceNum[1] == 1)
    mmFR = mmF1_mat;
  else
    mmFR = mmF2_mat;

  const int dg_np  = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS];
  const int dg_npf = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS + 1];
  const int findL = faceNum[0] * dg_npf;
  const int findR = faceNum[1] * dg_npf;
  const int *fmask  = FMASK_TK;
  const int *fmaskL = &fmask[faceNum[0] * dg_npf];
  const int *fmaskR = &fmask[faceNum[1] * dg_npf];

  DG_FP DL[DG_NP * DG_NP], DR[DG_NP * DG_NP];
  const DG_FP r_fact_0 = nx[0] * geof[0][RX_IND] + ny[0] * geof[0][RY_IND];
  const DG_FP s_fact_0 = nx[0] * geof[0][SX_IND] + ny[0] * geof[0][SY_IND];
  const DG_FP r_fact_1 = nx[1] * geof[1][RX_IND] + ny[1] * geof[1][RY_IND];
  const DG_FP s_fact_1 = nx[1] * geof[1][SX_IND] + ny[1] * geof[1][SY_IND];
  for(int i = 0; i < dg_np; i++) {
    for(int j = 0; j < dg_np; j++) {
      // int ind = i + j * DG_NP_N1;
      int ind = DG_MAT_IND(i, j, dg_np, dg_np);
      DL[ind] = r_fact_0 * dr_mat[ind] + s_fact_0 * ds_mat[ind];
      DR[ind] = r_fact_1 * dr_mat[ind] + s_fact_1 * ds_mat[ind];
    }
  }

  const DG_FP gtau = 2.0 * (*order + 1) * (*order + 2) * fmax(fscale[0], fscale[1]);

  op2_in_kernel_gemm(false, false, dg_np, dg_np, dg_np, -0.5 * sJ[0], mmFL, dg_np, DL, dg_np, 1.0, op1L, dg_np);
  op2_in_kernel_gemm(true, false, dg_np, dg_np, dg_np, -0.5 * sJ[0], DL, dg_np, mmFL, dg_np, 1.0, op1L, dg_np);

  const DG_FP tmp_constL = 0.5 * sJ[0] * gtau;
  for(int i = 0; i < dg_np; i++) {
    for(int j = 0; j < dg_np; j++) {
      // int op_ind = i + j * DG_NP_N1;
      int op_ind = DG_MAT_IND(i, j, dg_np, dg_np);
      op1L[op_ind] += tmp_constL * mmFL[op_ind];
    }
  }

  op2_in_kernel_gemm(false, false, dg_np, dg_np, dg_np, -0.5 * sJ[1], mmFR, dg_np, DR, dg_np, 1.0, op1R, dg_np);
  op2_in_kernel_gemm(true, false, dg_np, dg_np, dg_np, -0.5 * sJ[1], DR, dg_np, mmFR, dg_np, 1.0, op1R, dg_np);

  const DG_FP tmp_constR = 0.5 * sJ[1] * gtau;
  for(int i = 0; i < dg_np; i++) {
    for(int j = 0; j < dg_np; j++) {
      // int op_ind = i + j * DG_NP_N1;
      int op_ind = DG_MAT_IND(i, j, dg_np, dg_np);
      op1R[op_ind] += tmp_constR * mmFR[op_ind];
    }
  }
}
