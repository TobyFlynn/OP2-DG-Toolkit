#include "dg_utils.h"

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
