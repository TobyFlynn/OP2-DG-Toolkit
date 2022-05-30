#include "dg_utils.h"

#include <cmath>

// Calculate Jacobi polynomial of order N at points x
double DGUtils::jacobiP(const double x, const double alpha, const double beta,
                        const int N) {
  double gamma0 = pow(2.0, alpha + beta + 1.0) / (alpha + beta + 1.0) *
                  tgamma(alpha + 1.0) * tgamma(beta + 1.0) /
                  tgamma(alpha + beta + 1.0);
  double p_0 = 1.0 / sqrt(gamma0);

  // First base case
  if(N == 0) {
    return p_0;
  }

  double gamma1 = (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0) * gamma0;
  double p_1 = ((alpha + beta + 2.0) * x / 2.0 + (alpha - beta) / 2.0) /
               sqrt(gamma1);
  // Second base case
  if(N == 1) {
    return p_1;
  }

  double p_2 = 0.0;

  // Recurrence for N > 1
  double aOld = 2.0 / (2.0 + alpha + beta) * sqrt((alpha + 1.0) * (beta + 1.0) /
                (alpha + beta + 3.0));
  for(int i = 1; i < N; i++) {
    double h1 = 2.0 * i + alpha + beta;
    double aNew = 2.0 / (h1 + 2.0) * sqrt((i + 1.0) * (i + 1.0 + alpha + beta) *
                  (i + 1.0 + alpha) * (i + 1.0 + beta) / (h1 + 1.0) /
                  (h1 + 3.0));
    double bNew = -(alpha * alpha - beta * beta) / h1 / (h1 + 2.0);
    p_2 = 1.0 / aNew * (-aOld * p_0 + (x - bNew) * p_1);
    aOld = aNew;
    p_0 = p_1;
    p_1 = p_2;
  }

  return p_2;
}

double DGUtils::gradJacobiP(const double x, const double alpha,
                            const double beta, const int N) {
  if(N == 0) {
    return 0.0;
  } else {
    double fact = sqrt(N * (N + alpha + beta + 1.0));
    return fact * jacobiP(x, alpha + 1.0, beta + 1.0, N - 1);
  }
}

double DGUtils::grad2JacobiP(const double x, const double alpha,
                             const double beta, const int N) {
  if(N == 0 || N == 1) {
    return 0.0;
  } else {
    double fact = sqrt(N * (N + alpha + beta + 1.0));
    return fact * gradJacobiP(x, alpha + 1.0, beta + 1.0, N - 1);
  }
}

// Calculate 2D orthonomal poly on simplex of order i,j
double DGUtils::simplex2DP(const double a, const double b, const int i,
                           const int j) {
  double h1 = jacobiP(a, 0, 0, i);
  double h2 = jacobiP(b, 2 * i + 1, 0, j);
  return sqrt(2.0) * h1 * h2 * pow(1.0 - b, i);
}

// Calculate derivatives of modal basis on simplex
void DGUtils::gradSimplex2DP(const double a, const double b, const int i,
                             const int j, double &dr, double &ds) {
  double fa  = jacobiP(a, 0.0, 0.0, i);
  double gb  = jacobiP(b, 2.0 * i + 1.0, 0.0, j);
  double dfa = gradJacobiP(a, 0.0, 0.0, i);
  double dgb = gradJacobiP(b, 2.0 * i + 1.0, 0.0, j);

  // r derivative
  dr = dfa * gb;
  if(i > 0) {
    dr = dr * pow(0.5 * (1.0 - b), i - 1);
  }

  // s derivative
  ds = dfa * (gb * (0.5 * (1.0 + a)));
  if(i > 0) {
    ds = ds * pow(0.5 * (1.0 - b), i - 1);
  }

  double tmp = dgb * pow(0.5 * (1.0 - b), i);
  if(i > 0) {
    tmp = tmp - 0.5 * i * gb * pow(0.5 * (1.0 - b), i - 1);
  }
  ds = ds + fa * tmp;

  // Normalise
  dr = pow(2.0, i + 0.5) * dr;
  ds = pow(2.0, i + 0.5) * ds;
}

// Calculate simplexes for Hessian
void DGUtils::hessianSimplex2DP(const double a, const double b, const int i,
                                const int j, double &dr2, double &drs,
                                double &ds2) {
  double fa   = jacobiP(a, 0.0, 0.0, i);
  double gb   = jacobiP(b, 2.0 * i + 1.0, 0.0, j);
  double dfa  = gradJacobiP(a, 0.0, 0.0, i);
  double dgb  = gradJacobiP(b, 2.0 * i + 1.0, 0.0, j);
  double dfa2 = grad2JacobiP(a, 0.0, 0.0, i);
  double dgb2 = grad2JacobiP(b, 2.0 * i + 1.0, 0.0, j);

  // dr2
  dr2 = dfa2 * gb;
  if(i > 1) {
    dr2 = 4.0 * dr2 * pow(1.0 - b, i - 2);
  }

  // dsr
  // dsr = dfa % gb;
  // if(i > 1) {
  //   dsr = 2.0 * dsr % arma::pow(1.0 - b, i - 2);
  // }
  //
  // arma::vec tmp = arma::vec(a.n_elem, arma::fill::zeros);
  // if(i > 0) {
  //   tmp = dgb % arma::pow(1.0 - b, i - 1);
  // }
  // if(i > 1) {
  //   tmp = tmp - i * gb % arma::pow(1.0 - b, i - 2);
  // }
  // tmp = tmp % dfa * 2.0;
  // dsr = dsr + tmp;

  drs = 2.0 * (1.0 + a) * dfa2 * gb;
  if(i > 1) {
    drs = drs * pow(1.0 - b, i - 2);
  }
  double tmp = dfa * gb * 2.0;
  if(i > 1) {
    tmp = tmp * pow(1.0 - b, i - 2);
  }
  drs = drs + tmp;
  tmp = dgb;
  if(i > 0) {
    tmp = tmp * pow(1.0 - b, i - 1);
    if(i > 1) {
      tmp = tmp - i * gb * pow(1.0 - b, i - 2);
    } else {
      tmp = tmp - i * gb;
    }
  }
  tmp = tmp * dfa * 2.0;
  drs = drs + tmp;

  // ds2
  ds2 = dfa2 * gb * pow(1.0 + a, 2);
  if(i > 1) {
    ds2 = ds2 * pow(1.0 - b, i - 2);
  }

  double tmp2 = dgb2 * pow(1.0 - b, i);
  if(i > 0) {
    tmp2 = tmp2 - 2.0 * i * dgb * pow(1.0 - b, i - 1);
  }
  if(i > 1) {
    tmp2 = tmp2 + i * (i - 1) * gb * pow(1.0 - b, i - 2);
  }
  ds2 = ds2 + fa * tmp2;

  dr2 = pow(2.0, 0.5) * dr2;
  drs = pow(2.0, 0.5) * drs;
  ds2 = pow(2.0, 0.5) * ds2;
}
