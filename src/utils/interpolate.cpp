#include "dg_utils.h"

// Get the value at a point within a cell from modal values
double DGUtils::val_at_pt(const double r, const double s, const double *modal) {
  double a = -1.0;
  if(s != 1.0)
    a = 2.0 * (1.0 + r) / (1.0 - s) - 1.0;
  double b = s;

  double new_val = 0.0;
  int modal_ind = 0;
  for(int x_ = 0; x_ <= 3; x_++) {
    for(int y_ = 0; y_ <= 3 - x_; y_++) {
      new_val += modal[modal_ind++] * simplex2DP(a, b, x_, y_);
    }
  }

  return new_val;
}

// Get the gradient at a point within a cell from modal values
void DGUtils::grad_at_pt(const double r, const double s, const double *modal,
                double &dr, double &ds) {
  double a = -1.0;
  if(s != 1.0)
    a = 2.0 * (1.0 + r) / (1.0 - s) - 1.0;
  double b = s;

  dr = 0.0;
  ds = 0.0;
  int modal_ind = 0;
  for(int x_ = 0; x_ <= 3; x_++) {
    for(int y_ = 0; y_ <= 3 - x_; y_++) {
      double dr_tmp, ds_tmp;
      gradSimplex2DP(a, b, x_, y_, dr_tmp, ds_tmp);
      dr += modal[modal_ind] * dr_tmp;
      ds += modal[modal_ind++] * ds_tmp;
    }
  }
}

// Get the Hessian at a point within a cell from modal values
void DGUtils::hessian_at_pt(const double r, const double s, const double *modal,
                   double &dr2, double &drs, double &ds2) {
  double a = -1.0;
  if(s != 1.0)
    a = 2.0 * (1.0 + r) / (1.0 - s) - 1.0;
  double b = s;

  dr2 = 0.0;
  drs = 0.0;
  ds2 = 0.0;
  int modal_ind = 0;
  for(int x_ = 0; x_ <= 3; x_++) {
    for(int y_ = 0; y_ <= 3 - x_; y_++) {
      double dr2_tmp, drs_tmp, ds2_tmp;
      hessianSimplex2DP(a, b, x_, y_, dr2_tmp, drs_tmp, ds2_tmp);
      dr2 += modal[modal_ind] * dr2_tmp;
      drs += modal[modal_ind] * drs_tmp;
      ds2 += modal[modal_ind++] * ds2_tmp;
    }
  }
}
