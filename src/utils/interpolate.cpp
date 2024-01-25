#include "dg_utils.h"
/*
DG_FP DGUtils::val_at_pt_3d(const DG_FP r, const DG_FP s, const DG_FP t,
                            const DG_FP *modal, const int N) {
  DG_FP a, b, c;
  if(s + t != 0.0)
    a = 2.0 * (1.0 + r) / (-s - t) - 1.0;
  else
    a = -1.0;

  if(t != 1.0)
    b = 2.0 * (1.0 + s) / (1.0 - t) - 1.0;
  else
    b = -1.0;

  c = t;

  DG_FP new_val = 0.0;
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

void DGUtils::grad_at_pt_3d(const DG_FP r, const DG_FP s, const DG_FP t,
                            const DG_FP *modal, const int N, DG_FP &dr,
                            DG_FP &ds, DG_FP &dt) {
  DG_FP a, b, c;
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
*/
std::vector<DG_FP> DGUtils::val_at_pt_N_1_3d_get_simplexes(const std::vector<DG_FP> &r,
                      const std::vector<DG_FP> &s, const std::vector<DG_FP> &t,
                      const int N) {
  std::vector<DG_FP> a, b, c;
  for(int i = 0; i < r.size(); i++) {
    if(s[i] + t[i] != 0.0)
      a.push_back(2.0 * (1.0 + r[i]) / (-s[i] - t[i]) - 1.0);
    else
      a.push_back(-1.0);

    if(t[i] != 1.0)
      b.push_back(2.0 * (1.0 + s[i]) / (1.0 - t[i]) - 1.0);
    else
      b.push_back(-1.0);

    c.push_back(t[i]);
  }

  std::vector<DG_FP> result;

  for(int i = 0; i < r.size(); i++) {
    DG_FP new_val = 0.0;
    arma::vec a_(1), b_(1), c_(1), ans;
    a_(0) = a[i]; b_(0) = b[i]; c_(0) = c[i];
    for(int i = 0; i < N + 1; i++) {
      for(int j = 0; j < N - i + 1; j++) {
        for(int k = 0; k < N - i - j + 1; k++) {
          if(i + j + k < N) {
            ans = simplex3DP(a_, b_, c_, i, j, k);
            result.push_back(ans(0));
          } else {
            result.push_back(0.0);
          }
        }
      }
    }
  }

  return result;
}

DG_FP DGUtils::val_at_pt_N_1_3d(const DG_FP r, const DG_FP s, const DG_FP t,
                                const DG_FP *modal, const int N) {
  DG_FP a, b, c;
  if(s + t != 0.0)
    a = 2.0 * (1.0 + r) / (-s - t) - 1.0;
  else
    a = -1.0;

  if(t != 1.0)
    b = 2.0 * (1.0 + s) / (1.0 - t) - 1.0;
  else
    b = -1.0;

  c = t;

  DG_FP new_val = 0.0;
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

std::vector<DG_FP> DGUtils::val_at_pt_N_1_2d_get_simplexes(const std::vector<DG_FP> &r,
                      const std::vector<DG_FP> &s, const int N) {
  std::vector<DG_FP> result;
  for(int i = 0; i < r.size(); i++) {
    arma::vec r_vec(1), s_vec(1), a_vec(1), b_vec(1);
    r_vec(0) = r[i];
    s_vec(0) = s[i];
    rs2ab(r_vec, s_vec, a_vec, b_vec);

    for(int i = 0; i < N + 1; i++) {
      for(int j = 0; j < N + 1 - i; j++) {
        if(i + j < N) {
          arma::vec ans = simplex2DP(a_vec, b_vec, i, j);
          result.push_back(ans(0));
        } else {
          result.push_back(0.0);
        }
      }
    }
  }

  return result;
}


/*
 * Non arma::vec code
*/

double jacobiP_d(const double &x, const double alpha, const double beta, const int N) {
  double gamma0 = pow(2.0, alpha + beta + 1.0) / (alpha + beta + 1.0) *
                  tgamma(alpha + 1.0) * tgamma(beta + 1.0) /
                  tgamma(alpha + beta + 1.0);
  double pl_current = 1.0 / sqrt(gamma0);

  // First base case
  if(N == 0) {
    return pl_current;
  }
  double pl_old = pl_current;

  double gamma1 = (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0) * gamma0;
  pl_current = ((alpha + beta + 2.0) * x / 2.0 + (alpha - beta) / 2.0) /
                sqrt(gamma1);
  // Second base case
  if(N == 1) {
    return pl_current;
  }

  // Recurrence for N > 1
  double aOld = 2.0 / (2.0 + alpha + beta) * sqrt((alpha + 1.0) * (beta + 1.0) /
                (alpha + beta + 3.0));
  for(int i = 1; i < N; i++) {
    double h1 = 2.0 * i + alpha + beta;
    double aNew = 2.0 / (h1 + 2.0) * sqrt((i + 1.0) * (i + 1.0 + alpha + beta) *
                  (i + 1.0 + alpha) * (i + 1.0 + beta) / (h1 + 1.0) /
                  (h1 + 3.0));
    double bNew = -(alpha * alpha - beta * beta) / h1 / (h1 + 2.0);
    double pl_new = 1.0 / aNew * (-aOld * pl_old + (x - bNew) *
                    pl_current);
    pl_old = pl_current;
    pl_current = pl_new;
    aOld = aNew;
  }

  return pl_current;
}

double gradJacobiP_d(const double &x, const double alpha,
                     const double beta, const int N) {
  if(N == 0) {
    return 0.0;
  } else {
    double fact = sqrt(N * (N + alpha + beta + 1.0));
    return fact * jacobiP_d(x, alpha + 1.0, beta + 1.0, N - 1);
  }
}

double simplex3DP_d(const double &a, const double &b,
                    const double &c, const int i, const int j,
                    const int k) {
  double h1 = jacobiP_d(a, 0, 0, i);
  double h2 = jacobiP_d(b, 2 * i + 1, 0, j);
  double h3 = jacobiP_d(c, 2 * (i + j) + 2, 0, k);
  return 2.0 * sqrt(2.0) * h1 * h2 * pow(1.0 - b, i) * h3 * pow(1.0 - c, i + j);
}

void gradSimplex3DP_d(const double &a, const double &b,
                      const double &c, const int i, const int j,
                      const int k, double &dr, double &ds,
                      double &dt) {
  double fa  = jacobiP_d(a, 0, 0, i);
  double gb  = jacobiP_d(b, 2 * i + 1, 0, j);
  double hc  = jacobiP_d(c, 2 * (i + j) + 2, 0, k);
  double dfa = gradJacobiP_d(a, 0, 0, i);
  double dgb = gradJacobiP_d(b, 2 * i + 1, 0, j);
  double dhc = gradJacobiP_d(c, 2 * (i + j) + 2, 0, k);

  // r derivative
  dr = dfa * gb * hc;
  if(i > 0)
    dr = dr * pow(0.5 * (1.0 - b), i - 1);
  if(i + j > 0)
    dr = dr * pow(0.5 * (1.0 - c), i + j - 1);

  // s derivative
  ds = 0.5 * (1.0 + a) * dr;
  double tmp = dgb * pow(0.5 * (1.0 - b), i);
  if(i > 0)
    tmp = tmp + (-0.5 * i) * (gb * pow(0.5 * (1.0 - b), i - 1));
  if(i + j > 0)
    tmp = tmp * pow(0.5 * (1.0 - c), i + j - 1);
  tmp = fa * (tmp * hc);
  ds = ds + tmp;

  // t derivative
  dt = 0.5 * (1.0 + a) * dr + 0.5 * (1.0 + b) * tmp;
  tmp = dhc * pow(0.5 * (1.0 - c), i + j);
  if(i + j > 0)
    tmp = tmp - 0.5 * (i + j) * (hc * pow(0.5 * (1.0 - c), i + j - 1));
  tmp = fa * (gb * tmp);
  tmp = tmp * pow(0.5 * (1.0 - b), i);
  dt = dt + tmp;

  // normalise
  dr = dr * pow(2.0, 2 * i + j + 1.5);
  ds = ds * pow(2.0, 2 * i + j + 1.5);
  dt = dt * pow(2.0, 2 * i + j + 1.5);
}

DG_FP DGUtils::val_at_pt_3d(const DG_FP r, const DG_FP s, const DG_FP t,
                            const DG_FP *modal, const int N) {
  DG_FP a, b, c;
  if(s + t != 0.0)
    a = 2.0 * (1.0 + r) / (-s - t) - 1.0;
  else
    a = -1.0;

  if(t != 1.0)
    b = 2.0 * (1.0 + s) / (1.0 - t) - 1.0;
  else
    b = -1.0;

  c = t;

  DG_FP new_val = 0.0;
  int modal_ind = 0;
  for(int i = 0; i < N + 1; i++) {
    for(int j = 0; j < N - i + 1; j++) {
      for(int k = 0; k < N - i - j + 1; k++) {
        double ans = simplex3DP_d(a, b, c, i, j, k);
        new_val += modal[modal_ind++] * ans;
      }
    }
  }

  return new_val;
}

void DGUtils::grad_at_pt_3d(const DG_FP r, const DG_FP s, const DG_FP t,
                            const DG_FP *modal, const int N, DG_FP &dr,
                            DG_FP &ds, DG_FP &dt) {
  DG_FP a, b, c;
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
  for(int i = 0; i < N + 1; i++) {
    for(int j = 0; j < N - i + 1; j++) {
      for(int k = 0; k < N - i - j + 1; k++) {
        double dr_, ds_, dt_;
        gradSimplex3DP_d(a, b, c, i, j, k, dr_, ds_, dt_);
        dr += modal[modal_ind] * dr;
        ds += modal[modal_ind] * ds;
        dt += modal[modal_ind++] * dt;
      }
    }
  }
}
