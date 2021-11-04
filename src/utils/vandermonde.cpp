#include "dg_utils.h"

// Calculate 1D Vandermonde matrix
arma::mat DGUtils::vandermonde1D(const arma::vec &r, const int N) {
  arma::mat v1D(r.n_elem, N + 1);
  for(int j = 0; j < N + 1; j++) {
    v1D.col(j) = jacobiP(r, 0.0, 0.0, j);
  }
  return v1D;
}
