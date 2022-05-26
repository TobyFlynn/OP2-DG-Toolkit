#include "dg_utils.h"

// Get the value at a point within a cell from modal values
double DGUtils::val_at_pt(const double r, const double s, const double *modal) {
  double a = -1.0;
  if(s != 1.0)
    a = 2.0 * (1.0 + r) / (1.0 - s) - 1.0;
  double b = s;
  arma::vec a_(1);
  arma::vec b_(1);
  a_[0] = a;
  b_[0] = b;

  double new_val = 0.0;
  int modal_ind = 0;
  for(int x_ = 0; x_ <= 3; x_++) {
    for(int y_ = 0; y_ <= 3 - x_; y_++) {
      arma::vec res = simplex2DP(a_, b_, x_, y_);
      new_val += modal[modal_ind++] * res[0];
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
  arma::vec a_(1);
  arma::vec b_(1);
  a_[0] = a;
  b_[0] = b;

  dr = 0.0;
  ds = 0.0;
  int modal_ind = 0;
  arma::vec dr_v, ds_v;
  for(int x_ = 0; x_ <= 3; x_++) {
    for(int y_ = 0; y_ <= 3 - x_; y_++) {
      gradSimplex2DP(a_, b_, x_, y_, dr_v, ds_v);
      dr += modal[modal_ind] * dr_v[0];
      ds += modal[modal_ind++] * ds_v[0];
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
  arma::vec a_(1);
  arma::vec b_(1);
  a_[0] = a;
  b_[0] = b;

  dr2 = 0.0;
  drs = 0.0;
  ds2 = 0.0;
  int modal_ind = 0;
  arma::vec dr2_v, drs_v, ds2_v;
  for(int x_ = 0; x_ <= 3; x_++) {
    for(int y_ = 0; y_ <= 3 - x_; y_++) {
      hessianSimplex2DP(a_, b_, x_, y_, dr2_v, drs_v, ds2_v);
      dr2 += modal[modal_ind] * dr2_v[0];
      drs += modal[modal_ind] * drs_v[0];
      ds2 += modal[modal_ind++] * ds2_v[0];
    }
  }
}
