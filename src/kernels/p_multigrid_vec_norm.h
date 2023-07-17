inline void p_multigrid_vec_norm(const int *order, const DG_FP *vec,
                                 DG_FP *norm) {
  const int dg_np = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS];
  for(int i = 0; i < dg_np; i++) {
    *norm += vec[i] * vec[i];
  }
}
