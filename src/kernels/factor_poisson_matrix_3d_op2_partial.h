inline void factor_poisson_matrix_3d_op2_partial(const int **order, const int *faceNum,
                  const int *fmaskL_corrected, const int *fmaskR_corrected,
                  const DG_FP *nx, const DG_FP *ny, const DG_FP *nz,
                  const DG_FP *fscale, const DG_FP *sJ, const DG_FP **rx,
                  const DG_FP **sx, const DG_FP **tx, const DG_FP **ry,
                  const DG_FP **sy, const DG_FP **ty, const DG_FP **rz,
                  const DG_FP **sz, const DG_FP **tz, const DG_FP **factor,
                  DG_FP *op1L, DG_FP *op1R) {
  const int p = order[0][0];
  const DG_FP *dr_mat = &dg_Dr_kernel[(p - 1) * DG_NP * DG_NP];
  const DG_FP *ds_mat = &dg_Ds_kernel[(p - 1) * DG_NP * DG_NP];
  const DG_FP *dt_mat = &dg_Dt_kernel[(p - 1) * DG_NP * DG_NP];
  const DG_FP *mmF0_mat = &dg_MM_F0_kernel[(p - 1) * DG_NP * DG_NP];
  const DG_FP *mmF1_mat = &dg_MM_F1_kernel[(p - 1) * DG_NP * DG_NP];
  const DG_FP *mmF2_mat = &dg_MM_F2_kernel[(p - 1) * DG_NP * DG_NP];
  const DG_FP *mmF3_mat = &dg_MM_F3_kernel[(p - 1) * DG_NP * DG_NP];
  const int dg_np  = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS];
  const int dg_npf = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS + 1];

  const DG_FP tau_order = (DG_FP) p; // (DG_FP) DG_ORDER;

  const DG_FP *mmFL, *mmFR;
  if(faceNum[0] == 0)
    mmFL = mmF0_mat;
  else if(faceNum[0] == 1)
    mmFL = mmF1_mat;
  else if(faceNum[0] == 2)
    mmFL = mmF2_mat;
  else
    mmFL = mmF3_mat;

  if(faceNum[1] == 0)
    mmFR = mmF0_mat;
  else if(faceNum[1] == 1)
    mmFR = mmF1_mat;
  else if(faceNum[1] == 2)
    mmFR = mmF2_mat;
  else
    mmFR = mmF3_mat;

  const int findL = faceNum[0] * dg_npf;
  const int findR = faceNum[1] * dg_npf;
  const int *fmask  = &FMASK_TK[(p - 1) * 4 * DG_NPF];
  const int *fmaskL = &fmask[faceNum[0] * dg_npf];
  const int *fmaskR = &fmask[faceNum[1] * dg_npf];

  DG_FP DL[DG_NP * DG_NP], DR[DG_NP * DG_NP];
  const DG_FP r_fact_0 = nx[0] * rx[0][0] + ny[0] * ry[0][0] + nz[0] * rz[0][0];
  const DG_FP s_fact_0 = nx[0] * sx[0][0] + ny[0] * sy[0][0] + nz[0] * sz[0][0];
  const DG_FP t_fact_0 = nx[0] * tx[0][0] + ny[0] * ty[0][0] + nz[0] * tz[0][0];
  const DG_FP r_fact_1 = nx[1] * rx[1][0] + ny[1] * ry[1][0] + nz[1] * rz[1][0];
  const DG_FP s_fact_1 = nx[1] * sx[1][0] + ny[1] * sy[1][0] + nz[1] * sz[1][0];
  const DG_FP t_fact_1 = nx[1] * tx[1][0] + ny[1] * ty[1][0] + nz[1] * tz[1][0];
  for(int i = 0; i < dg_np; i++) {
    for(int j = 0; j < dg_np; j++) {
      // int ind = i + j * dg_np;
      int ind = DG_MAT_IND(i, j, dg_np, dg_np);

      DL[ind] = r_fact_0 * dr_mat[ind] + s_fact_0 * ds_mat[ind] + t_fact_0 * dt_mat[ind];
      DL[ind] *= factor[0][i];

      DR[ind] = r_fact_1 * dr_mat[ind] + s_fact_1 * ds_mat[ind] + t_fact_1 * dt_mat[ind];
      DR[ind] *= factor[1][i];
    }
  }

  DG_FP gtau = 0.0;
  for(int i = 0; i < dg_npf; i++) {
    DG_FP tmp = 2.0 * (tau_order + 1) * (tau_order + 2) * fmax(fscale[0] * factor[0][fmaskL[i]], fscale[1] * factor[1][fmaskR_corrected[i]]);
    gtau = fmax(gtau, tmp);
  }

  op2_in_kernel_gemm(false, false, dg_np, dg_np, dg_np, -0.5 * sJ[0], mmFL, dg_np, DL, dg_np, 1.0, op1L, dg_np);
  op2_in_kernel_gemm(true, false, dg_np, dg_np, dg_np, -0.5 * sJ[0], DL, dg_np, mmFL, dg_np, 1.0, op1L, dg_np);

  const DG_FP tmp_constL = 0.5 * sJ[0] * gtau;
  for(int i = 0; i < dg_np; i++) {
    for(int j = 0; j < dg_np; j++) {
      // int op_ind = i + j * dg_np;
      int op_ind = DG_MAT_IND(i, j, dg_np, dg_np);
      op1L[op_ind] += tmp_constL * mmFL[op_ind];
    }
  }

  op2_in_kernel_gemm(false, false, dg_np, dg_np, dg_np, -0.5 * sJ[1], mmFR, dg_np, DR, dg_np, 1.0, op1R, dg_np);
  op2_in_kernel_gemm(true, false, dg_np, dg_np, dg_np, -0.5 * sJ[1], DR, dg_np, mmFR, dg_np, 1.0, op1R, dg_np);

  const DG_FP tmp_constR = 0.5 * sJ[1] * gtau;
  for(int i = 0; i < dg_np; i++) {
    for(int j = 0; j < dg_np; j++) {
      // int op_ind = i + j * dg_np;
      int op_ind = DG_MAT_IND(i, j, dg_np, dg_np);
      op1R[op_ind] += tmp_constR * mmFR[op_ind];
    }
  }
}
