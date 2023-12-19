inline void factor_poisson_matrix_2d_mm_block_diag(const int *order,
                const DG_FP *factor, const DG_FP *geof, DG_FP *op1) {
  const DG_FP *mass_mat = &dg_Mass_kernel[(*order - 1) * DG_NP * DG_NP];
  const int dg_np = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS];

  const DG_FP J = geof[J_IND];
  for(int i = 0; i < dg_np; i++) {
    for(int j = 0; j < dg_np; j++) {
      // int op_ind = i + j * dg_np;
      int op_ind = DG_MAT_IND(i, j, dg_np, dg_np);
      op1[op_ind] += factor[i] * J * mass_mat[op_ind];
    }
  }
}