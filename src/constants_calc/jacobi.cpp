#include "dg_constants.h"

#include <cmath>

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

void DGConstants::jacobiGQ(double alpha, double beta, const int n,
                           std::vector<double> &x, std::vector<double> &w) {
  if(n == 0) {
    x.clear(); w.clear();
    x.push_back(-(alpha - beta) / (alpha + beta + 2.0));
    w.push_back(2.0);
    return;
  }

  std::vector<double> h1(n + 1);
  for(int i = 0; i < n + 1; i++) {
    h1[i] = 2.0 * i + alpha + beta;
  }

  arma::vec j1(n + 1);
  for(int i = 0; i < n + 1; i++) {
    j1[i] = -0.5 * (alpha * alpha - beta * beta) / (h1[i] + 2.0) / h1[i];
  }

  arma::vec j2(n);
  for(int i = 0; i < n; i++) {
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
  x.resize(eigVal.n_elem); w.resize(eigVal.n_elem);

  for(int i = 0; i < eigVal.n_elem; i++) {
    x[i] = eigVal(i);
    w[i] = eigVec(0, i) * eigVec(0, i) * pow(2.0, alpha + beta + 1.0) /
           (alpha + beta + 1.0) * tgamma(alpha + 1.0) * tgamma(beta + 1.0) /
           tgamma(alpha + beta + 1.0);
  }
}

std::vector<double> DGConstants::jacobiGL(double alpha, double beta, const int n) {
  std::vector<double> x(n + 1);
  if(n == 1) {
    x[0] = -1.0;
    x[1] = 1.0;
    return x;
  }

  std::vector<double> w;
  jacobiGQ(alpha + 1.0, beta + 1.0, n - 2, x, w);
  x.insert(x.begin(), -1.0);
  x.push_back(1.0);
  return x;
}
