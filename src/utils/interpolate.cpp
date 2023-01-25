#include "dg_utils.h"

double DGUtils::val_at_pt_3d(const double r, const double s, const double t,
                             const double *modal, const int N) {
  double a, b, c;
  if(s + t != 0.0)
    a = 2.0 * (1.0 + r) / (-s - t) - 1.0;
  else
    a = -1.0;

  if(t != 1.0)
    b = 2.0 * (1.0 + s) / (1.0 - t) - 1.0;
  else
    b = -1.0;

  c = t;

  double new_val = 0.0;
  int modal_ind = 0;
  arma::vec a_(1), b_(1), c_(1), ans;
  a_(0) = a; b_(0) = b; c_(0) = c;
  for(int i = 0; i < N + 1; i++) {
    for(int j = 0; j < N - i + 1; j++) {
      for(int k = 0; k < N - i - j + 1; k++) {
        ans = simplex3DP(a_, b_, c_, i, j, k);
        new_val += modal[modal_ind++] * ans(0);
      }
    }
  }

  return new_val;
}

void DGUtils::grad_at_pt_3d(const double r, const double s, const double t,
                            const double *modal, const int N, double &dr,
                            double &ds, double &dt) {
  double a, b, c;
  if(s + t != 0.0)
    a = 2.0 * (1.0 + r) / (-s - t) - 1.0;
  else
    a = -1.0;

  if(t != 1.0)
    b = 2.0 * (1.0 + s) / (1.0 - t) - 1.0;
  else
    b = -1.0;

  c = t;

  dr = 0.0; ds = 0.0; dt = 0.0;
  int modal_ind = 0;
  arma::vec a_(1), b_(1), c_(1), dr_, ds_, dt_;
  a_(0) = a; b_(0) = b; c_(0) = c;
  for(int i = 0; i < N + 1; i++) {
    for(int j = 0; j < N - i + 1; j++) {
      for(int k = 0; k < N - i - j + 1; k++) {
        gradSimplex3DP(a_, b_, c_, i, j, k, dr_, ds_, dt_);
        dr += modal[modal_ind] * dr_(0);
        ds += modal[modal_ind] * ds_(0);
        dt += modal[modal_ind++] * dt_(0);
      }
    }
  }
}

double DGUtils::val_at_pt_N_1_3d(const double r, const double s, const double t,
                                 const double *modal, const int N) {
  double a, b, c;
  if(s + t != 0.0)
    a = 2.0 * (1.0 + r) / (-s - t) - 1.0;
  else
    a = -1.0;

  if(t != 1.0)
    b = 2.0 * (1.0 + s) / (1.0 - t) - 1.0;
  else
    b = -1.0;

  c = t;

  double new_val = 0.0;
  int modal_ind = 0;
  arma::vec a_(1), b_(1), c_(1), ans;
  a_(0) = a; b_(0) = b; c_(0) = c;
  for(int i = 0; i < N + 1; i++) {
    for(int j = 0; j < N - i + 1; j++) {
      for(int k = 0; k < N - i - j + 1; k++) {
        if(i + j + k < N) {
          ans = simplex3DP(a_, b_, c_, i, j, k);
          new_val += modal[modal_ind] * ans(0);
        }
        modal_ind++;
      }
    }
  }

  return new_val;
}