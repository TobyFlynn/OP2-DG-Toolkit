inline void zero_npf_3_tk(DG_FP * __restrict__ x, DG_FP * __restrict__ y, DG_FP * __restrict__ z) {
  for(int i = 0; i < DG_NUM_FACES * DG_NPF; i++) {
    x[i] = 0.0;
    y[i] = 0.0;
    z[i] = 0.0;
  }
}
