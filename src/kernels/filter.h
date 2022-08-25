inline void filter(const int *p, double *modal) {
  const int cutoff_N = 1;
  const double alpha = 36.0;
  const double s = 8.0;
  
  int ind = 0;
  for(int i = 0; i <= *p; i++) {
    for(int j = 0; j <= *p - i; j++) {
      if(i + j >= cutoff_N) {
        const double n = (double)(i + j - cutoff_N) / (double)(*p - cutoff_N);
        modal[ind] *= exp(-alpha * pow(n, s));
      }
      ind++;
    }
  }
}