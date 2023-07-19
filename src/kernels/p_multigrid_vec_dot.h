inline void p_multigrid_vec_dot(const int *order, const float *vec0,
                                const float *vec1, float *dot) {
  const int dg_np = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS];

  for(int i = 0; i < dg_np; i++) {
    *dot += vec0[i] * vec1[i];
  }
}
