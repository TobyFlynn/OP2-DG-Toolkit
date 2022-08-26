inline void filter(const int *p, double *modal) {
  const double PI = 3.141592653589793238463;
  const int cutoff_N = 1;
  const double alpha = 36.0;
  const double s = 8.0;
  
  double q[DG_ORDER + 1];
  for(int i = 0; i < DG_ORDER + 1; i++) {
    q[i] = 0.0;
  }

  int ind = 0;
  for(int i = 0; i <= *p; i++) {
    for(int j = 0; j <= *p - i; j++) {
      q[i + j] += modal[ind] * modal[ind];
      ind++;
    }
  }

  for(int i = 0; i < DG_ORDER + 1; i++) {
    q[i] = sqrt(q[i]);
  }

  q[*p] = fmax(q[*p], q[*p - 1]);
  for(int i = *p - 1; i > 0; i--) {
    q[i] = fmax(q[i], q[i + 1]);
  }

  double sum1 = 0.0;
  double sum2 = 0.0;
  double sum3 = 0.0;
  double sum4 = 0.0;
  for(int i = 1; i < DG_ORDER + 1; i++) {
    double logx = logf(i);
    double logq = logf(q[i]);
    sum1 += logq * logx;
    sum2 += logq;
    sum3 += logx;
    sum4 += logx * logx;
  }
  double b = (DG_ORDER * sum1 - sum2 * sum3) / (DG_ORDER * sum4 - sum3 * sum3);
  double a = (sum2 - b * sum3) / (double)DG_ORDER;
  double decay_exponent = -b;

  double filter_strength = 0.0;
  if(decay_exponent < 1.0) {
    filter_strength = 1.0;
  } else if(decay_exponent <= 3.0) {
    filter_strength = 0.5 * (1.0 + sin(- PI * (decay_exponent - 2.0) / 2.0));
  }

  ind = 0;
  for(int i = 0; i <= *p; i++) {
    for(int j = 0; j <= *p - i; j++) {
      if(i + j >= cutoff_N) {
        const double n = (double)(i + j - cutoff_N) / (double)(*p - cutoff_N);
        modal[ind] *= exp(-filter_strength * alpha * pow(n, s));
      }
      ind++;
    }
  }
}