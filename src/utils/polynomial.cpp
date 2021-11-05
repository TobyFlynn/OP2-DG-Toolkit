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

// Calculate Jacobi polynomial of order N at points x
arma::vec DGUtils::jacobiP(const arma::vec &x, const double alpha,
                           const double beta, const int N) {
  arma::mat pl(x.n_elem, N + 1);
  double gamma0 = pow(2.0, alpha + beta + 1.0) / (alpha + beta + 1.0) *
                  tgamma(alpha + 1.0) * tgamma(beta + 1.0) /
                  tgamma(alpha + beta + 1.0);
  arma::vec col0(x.n_elem);
  col0.fill(1.0 / sqrt(gamma0));

  // First base case
  if(N == 0) {
    return col0;
  }
  pl.col(0) = col0;

  double gamma1 = (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0) * gamma0;
  pl.col(1) = ((alpha + beta + 2.0) * x / 2.0 + (alpha - beta) / 2.0) /
              sqrt(gamma1);
  // Second base case
  if(N == 1) {
    return pl.col(1);
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
    pl.col(i + 1) = 1.0 / aNew * (-aOld * pl.col(i - 1) + (x - bNew) %
                    pl.col(i));
    aOld = aNew;
  }

  return pl.col(N);
}

arma::vec DGUtils::simplex2DP(const arma::vec &a, const arma::vec &b,
                              const int i, const int j) {
  arma::vec h1 = jacobiP(a, 0, 0, i);
  arma::vec h2 = jacobiP(b, 2 * i + 1, 0, j);
  return sqrt(2.0) * h1 % h2 % arma::pow(1.0 - b, i);
}
