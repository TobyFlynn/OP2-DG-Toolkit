inline void inv_mass(const int *p, const DG_FP *matrix, const DG_FP *J, DG_FP *x) {
  // Get constants
  const int dg_np  = DG_CONSTANTS[(*p - 1) * 5];
  const DG_FP *mat = &matrix[(*p - 1) * DG_NP * DG_NP];

  DG_FP tmp[DG_NP];
  for(int i = 0; i < dg_np; i++) {
    tmp[i] = 0.0;
    for(int j = 0; j < dg_np; j++) {
      // int ind = i + j * DG_NP;
      int ind = DG_MAT_IND(i, j, dg_np, dg_np);
      tmp[i] += mat[ind] * x[j];
    }
  }

  for(int i = 0; i < dg_np; i++) {
    x[i] = tmp[i] / *J;
  }
}
