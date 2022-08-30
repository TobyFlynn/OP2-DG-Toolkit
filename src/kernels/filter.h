inline void filter(const int *p, const double *v_g, const double *mass_g, 
                   const double *J, const double *u, double *modal, double *shock) {
  if(*p < 2)
    return;
  const double PI = 3.141592653589793238463;
  const int cutoff_N = 0;
  const double alpha = 36.0;
  const double s = 8.0;
  const int dg_np = DG_CONSTANTS[(*p - 1) * 5];
  const double *v_mat = &v_g[(*p - 1) * DG_NP * DG_NP];
  const double *m_mat = &mass_g[(*p - 1) * DG_NP * DG_NP];
  
  int ind = 0;
  double p_modal[dg_np];
  for(int i = 0; i <= *p; i++) {
    for(int j = 0; j <= *p - i; j++) {
      if(i + j == *p) {
        p_modal[ind] = modal[ind];
      } else {
        p_modal[ind] = 0.0;
      }
      ind++;
    }
  }

  double p_u[dg_np];
  for(int i = 0; i < dg_np; i++) {
    p_u[i] = 0.0;
    for(int j = 0; j < dg_np; j++) {
      int ind_mat = i + j * dg_np;
      p_u[i] += v_mat[ind_mat] * p_modal[j];
    }
  }

  double mass_u = 0.0;
  double mass_p_u = 0.0;
  for(int i = 0; i < dg_np; i++) {
    double mass_w = 0.0;
    double u_c = 0.0;
    double p_u_c = 0.0;
    for(int j = 0; j < dg_np; j++) {
      int ind_mat = i + j * dg_np;
      // int ind_mat = i * dg_np + j;
      // mass_w += m_mat[ind_mat];
      // mass_u[i] += m_mat[ind_mat] * u[j];
      // mass_p_u[i] += m_mat[ind_mat] * p_u[j];
      u_c += m_mat[ind_mat] * J[j] * u[j];
      p_u_c += m_mat[ind_mat] * J[j] * p_u[j];
    }
    // mass_u += J[i] * u[i] * u[i] * mass_w;
    // mass_p_u += J[i] * p_u[i] * p_u[i] * mass_w;
    mass_u += u[i] * u_c;
    mass_p_u += p_u[i] * p_u_c;
  }

  double filter_strength[dg_np];
  double s_ref = -4.0 * log((double)*p);
  double max_filter = 1.0;
  double k = 1.0;
  double total = 0.0;
  for(int i = 0; i < dg_np; i++) {
    double s = log(mass_p_u / mass_u);
    if(s < s_ref - max_filter) {
      filter_strength[i] = 0.0;
    } else if(s < s_ref + max_filter) {
      filter_strength[i] = (max_filter / 2.0) * (1.0 + sin(PI * (s - s_ref) / (2.0 * k)));
    } else {
      filter_strength[i] = max_filter;
    }
  }

  ind = 0;
  for(int i = 0; i <= *p; i++) {
    for(int j = 0; j <= *p - i; j++) {
      if(i + j >= cutoff_N) {
        const double n = (double)(i + j - cutoff_N) / (double)(*p - cutoff_N);
        modal[ind] *= exp(-filter_strength[i] * alpha * pow(n, s));
        // modal[ind] *= exp(-alpha * pow(n, s));
        shock[ind] = filter_strength[i];
      }
      ind++;
    }
  }
}