inline void zero_npf(double *a) {
  for(int i = 0; i < DG_NUM_FACES * DG_NPF; i++) {
    a[i] = 0.0;
  }
}