#include "dg_utils.h"

// Calculate 1D Vandermonde matrix
arma::mat DGUtils::vandermonde1D(const arma::vec &r, const int N) {
  arma::mat v1D(r.n_elem, N + 1);
  for(int j = 0; j < N + 1; j++) {
    v1D.col(j) = jacobiP(r, 0.0, 0.0, j);
  }
  return v1D;
}

// Calculate 2D Vandermonde matrix
arma::mat DGUtils::vandermonde2D(const arma::vec &r, const arma::vec &s,
                                 const int N) {
  // Transfer to a-b coordinates
  arma::vec a, b;
  rs2ab(r, s, a, b);

  // Build matrix
  arma::mat v2D(r.n_elem, (N + 1) * (N + 2) / 2);
  int col = 0;
  for(int i = 0; i < N + 1; i++) {
    for(int j = 0; j < N + 1 - i; j++) {
      v2D.col(col) = simplex2DP(a, b, i, j);
      col++;
    }
  }

  return v2D;
}

// Vandermonde matrix for gradient of modal basis
void DGUtils::gradVandermonde2D(const arma::vec &r, const arma::vec &s,
                                const int N, arma::mat &vDr, arma::mat &vDs) {
  vDr.set_size(r.n_elem, (N + 1) * (N + 2) / 2);
  vDs.set_size(r.n_elem, (N + 1) * (N + 2) / 2);

  arma::vec a, b;
  rs2ab(r, s, a, b);

  int col = 0;
  for(int i = 0; i < N + 1; i++) {
    for(int j = 0; j < N + 1 - i; j++) {
      arma::vec dr, ds;
      gradSimplex2DP(a, b, i, j, dr, ds);
      vDr.col(col) = dr;
      vDs.col(col) = ds;
      col++;
    }
  }
}
