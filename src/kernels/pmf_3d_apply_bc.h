inline void pmf_3d_apply_bc(const int *p, const int *faceNum, const int *bc_type,
                            const DG_FP *nx, const DG_FP *ny, const DG_FP *nz, const DG_FP *fscale,
                            const DG_FP *sJ, const DG_FP *geof, const DG_FP *bc, DG_FP *rhs) {
  const DG_FP *dr_mat = &dg_Dr_kernel[(*p - 1) * DG_NP * DG_NP];
  const DG_FP *ds_mat = &dg_Ds_kernel[(*p - 1) * DG_NP * DG_NP];
  const DG_FP *dt_mat = &dg_Dt_kernel[(*p - 1) * DG_NP * DG_NP];
  const DG_FP *mmF0_mat = &dg_MM_F0_kernel[(*p - 1) * DG_NP * DG_NP];
  const DG_FP *mmF1_mat = &dg_MM_F1_kernel[(*p - 1) * DG_NP * DG_NP];
  const DG_FP *mmF2_mat = &dg_MM_F2_kernel[(*p - 1) * DG_NP * DG_NP];
  const DG_FP *mmF3_mat = &dg_MM_F3_kernel[(*p - 1) * DG_NP * DG_NP];
  const int dg_np  = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS];
  const int dg_npf = DG_CONSTANTS_TK[(*p - 1) * DG_NUM_CONSTANTS + 1];

  const DG_FP *mmF;
  if(*faceNum == 0)
    mmF = mmF0_mat;
  else if(*faceNum == 1)
    mmF = mmF1_mat;
  else if(*faceNum == 2)
    mmF = mmF2_mat;
  else
    mmF = mmF3_mat;

  const int find = *faceNum * dg_npf;
  const int *fmask  = &FMASK_TK[(*p - 1) * DG_NUM_FACES * DG_NPF];
  const int *fmaskB = &fmask[*faceNum * dg_npf];

  if(*bc_type == 0) {
    // Dirichlet
    DG_FP D[DG_NP * DG_NP];
    for(int i = 0; i < dg_np; i++) {
      for(int j = 0; j < dg_np; j++) {
        // int ind = i + j * dg_np;
        int ind = DG_MAT_IND(i, j, dg_np, dg_np);

        D[ind] = *nx * (geof[RX_IND] * dr_mat[ind] + geof[SX_IND] * ds_mat[ind] + geof[TX_IND] * dt_mat[ind]);
        D[ind] += *ny * (geof[RY_IND] * dr_mat[ind] + geof[SY_IND] * ds_mat[ind] + geof[TY_IND] * dt_mat[ind]);
        D[ind] += *nz * (geof[RZ_IND] * dr_mat[ind] + geof[SZ_IND] * ds_mat[ind] + geof[TZ_IND] * dt_mat[ind]);
      }
    }

    const DG_FP gtau = 2.0 * (DG_ORDER + 1) * (DG_ORDER + 2) * *fscale;

    DG_FP tmp[DG_NP];
    for(int i = 0; i < dg_np; i++) {
      tmp[i] = 0.0;
      for(int j = 0; j < dg_npf; j++) {
        // int mm_ind = i + fmaskB[j] * dg_np;
        int mm_ind = DG_MAT_IND(i, fmaskB[j], dg_np, dg_np);
        rhs[i] += gtau * *sJ * mmF[mm_ind] * bc[j];
        tmp[i] += *sJ * mmF[mm_ind] * bc[j];
      }
    }

    for(int i = 0; i < dg_np; i++) {
      for(int j = 0; j < dg_np; j++) {
        // int mm_ind = i + fmaskB[j] * dg_np;
        int d_ind = DG_MAT_IND(j, i, dg_np, dg_np);
        rhs[i] += -D[d_ind] * tmp[j];
      }
    }
  } else {
    // Neumann
    for(int i = 0; i < dg_np; i++) {
      for(int j = 0; j < dg_npf; j++) {
        // int mm_ind = i + fmaskB[j] * dg_np;
        int mm_ind = DG_MAT_IND(i, fmaskB[j], dg_np, dg_np);
        rhs[i] += *sJ * mmF[mm_ind] * bc[j];
      }
    }
  }
}
