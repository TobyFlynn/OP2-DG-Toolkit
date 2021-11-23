inline void cub_mm_init(const int *p, const double *matrix, const double *tmp,
                        double *mm) {
  // Get constants for this element's order
  const int dg_np     = DG_CONSTANTS[(*p - 1) * 5];
  const int dg_cub_np = DG_CONSTANTS[(*p - 1) * 5 + 2];
  const double *cubV  = &matrix[(*p - 1) * DG_CUB_NP * DG_NP];

  for(int i = 0; i < dg_np; i++) {
    for(int j = 0; j < dg_np; j++) {
      int mmInd = i + j * dg_np;
      mm[mmInd] = 0.0;
      for(int k = 0; k < dg_cub_np; k++) {
        int aInd = i * dg_cub_np + k;
        int bInd = j * dg_cub_np + k;
        mm[mmInd] += cubV[aInd] * tmp[bInd];
      }
    }
  }
}
