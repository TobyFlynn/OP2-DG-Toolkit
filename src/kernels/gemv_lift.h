inline void gemv_lift(const int *p, const DG_FP *alpha, const DG_FP *beta,
                      const DG_FP *matrix, const DG_FP *x, DG_FP *y) {
  // Get constants
  const int dg_np    = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS];
  const int dg_nfp   = DG_CONSTANTS[(*p - 1) * DG_NUM_CONSTANTS + 1];
  const DG_FP *lift = &matrix[(*p - 1) * DG_NUM_FACES * DG_NPF * DG_NP];

  for(int i = 0; i < dg_np; i++) {
    y[i] *= *beta;
    for(int j = 0; j < DG_NUM_FACES * dg_nfp; j++) {
      // int ind = i + j * dg_np;
      int ind = DG_MAT_IND(i, j, dg_np, DG_NUM_FACES * dg_nfp);
      y[i] += *alpha * lift[ind] * x[j];
    }
  }
}
