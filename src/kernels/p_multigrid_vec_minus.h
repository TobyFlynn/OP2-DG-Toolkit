inline void p_multigrid_vec_minus(const int *order, const float *vec0,
                                  float *vec1, const float *dot) {
  const int dg_np = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS];

  for(int i = 0; i < dg_np; i++) {
    vec1[i] -= *dot * vec0[i];
  }
}
