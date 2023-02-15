inline void cub_mm_init(const int *p, const DG_FP *matrix, const DG_FP *tmp,
                        DG_FP *mm) {
  // Get constants for this element's order
  const int dg_np     = DG_CONSTANTS[(*p - 1) * 5];
  const int dg_cub_np = DG_CONSTANTS[(*p - 1) * 5 + 2];
  const DG_FP *cubV  = &matrix[(*p - 1) * DG_CUB_NP * DG_NP];

  for(int i = 0; i < dg_np; i++) {
    for(int j = 0; j < dg_np; j++) {
      // int mmInd = i + j * dg_np;
      int mmInd = DG_MAT_IND(i, j, dg_np, dg_np);
      mm[mmInd] = 0.0;
      for(int k = 0; k < dg_cub_np; k++) {
        // int aInd = i * dg_cub_np + k;
        int aInd = DG_MAT_IND(k, i, dg_cub_np, dg_np);
        // int bInd = j * dg_cub_np + k;
        int bInd = DG_MAT_IND(k, j, dg_cub_np, dg_np);
        mm[mmInd] += cubV[aInd] * tmp[bInd];
      }
    }
  }
}
