#include "dg_utils.h"

#include <cmath>

void DGUtils::jacobiGQ(const double alpha, const double beta, const int N,
                       arma::vec &x, arma::vec &w) {
  x.reset(); w.reset();
  if(N == 0) {
    x.set_size(1); w.set_size(1);
    x[0] = -(alpha - beta) / (alpha + beta + 2.0);
    w[0] = 2.0;
    return;
  }

  arma::vec h1(N + 1);
  for(int i = 0; i < N + 1; i++) {
    h1[i] = 2.0 * i + alpha + beta;
  }

  arma::vec j1(N + 1);
  for(int i = 0; i < N + 1; i++) {
    j1[i] = -0.5 * (alpha * alpha - beta * beta) / (h1[i] + 2.0) / h1[i];
  }

  arma::vec j2(N);
  for(int i = 0; i < N; i++) {
    j2[i]  = 2.0 / (h1[i] + 2.0);
    j2[i] *= sqrt((i + 1.0) * ((i + 1.0) + alpha + beta) * ((i + 1.0) + alpha) *
             ((i + 1.0) + beta) / (h1[i] + 1.0) / (h1[i] + 3.0));
  }
  arma::mat j = diagmat(j1) + diagmat(j2, 1);
  if(alpha + beta < 1e-15)
    j(0,0) = 0.0;

  j = j + j.t();

  arma::vec eigVal;
  arma::mat eigVec;
  arma::eig_sym(eigVal, eigVec, j);

  x.clear(); w.clear();
  x.set_size(eigVal.n_elem); w.set_size(eigVal.n_elem);

  for(int i = 0; i < eigVal.n_elem; i++) {
    x[i] = eigVal(i);
    w[i] = eigVec(0, i) * eigVec(0, i) * pow(2.0, alpha + beta + 1.0) /
           (alpha + beta + 1.0) * tgamma(alpha + 1.0) * tgamma(beta + 1.0) /
           tgamma(alpha + beta + 1.0);
  }
}

arma::vec DGUtils::jacobiGL(const double alpha, const double beta,
                            const int N) {
  arma::vec x(N + 1);
  if(N == 1) {
    x[0] = -1.0;
    x[1] = 1.0;
    return x;
  }

  arma::vec w;
  jacobiGQ(alpha + 1.0, beta + 1.0, N - 2, x, w);
  std::vector<double> x_tmp = arma::conv_to<std::vector<double>>::from(x);
  x_tmp.insert(x_tmp.begin(), -1.0);
  x_tmp.push_back(1.0);
  return arma::vec(x_tmp);
}
