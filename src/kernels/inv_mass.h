inline void inv_mass(const double *matrix, const double *J, double *x) {
  // Get constants
  // const int dg_np   = DG_CONSTANTS[(*p - 1) * 5];
  const double *mat = &matrix[(DG_ORDER - 1) * DG_NP * DG_NP];

  double tmp[DG_NP];
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