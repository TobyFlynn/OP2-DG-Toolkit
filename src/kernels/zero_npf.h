inline void zero_npf(DG_FP *a) {
  for(int i = 0; i < DG_NUM_FACES * DG_NPF; i++) {
    a[i] = 0.0;
  }
}