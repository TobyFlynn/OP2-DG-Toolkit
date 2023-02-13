inline void inv_mass(const DG_FP *matrix, const DG_FP *J, DG_FP *x) {
  // Get constants
  // const int dg_np   = DG_CONSTANTS[(*p - 1) * 5];
  const DG_FP *mat = &matrix[(DG_ORDER - 1) * DG_NP * DG_NP];

  DG_FP tmp[DG_NP];
  for(int i = 0; i < DG_NP; i++) {
    tmp[i] = 0.0;
    for(int j = 0; j < DG_NP; j++) {
      int ind = i + j * DG_NP;
      tmp[i] += mat[ind] * x[j];
    }
  }

  for(int i = 0; i < DG_NP; i++) {
    x[i] = tmp[i] / *J;
  }
}