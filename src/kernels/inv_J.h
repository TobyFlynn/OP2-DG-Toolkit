inline void inv_J(const int *p, const double *J, const double *tmp, double *u) {
  // Get constants for this element's order
  const int dg_np = DG_CONSTANTS[(*p - 1) * 5];
  for(int i = 0; i < dg_np; i++) {
    u[i] = tmp[i] / J[i];
  }
}
