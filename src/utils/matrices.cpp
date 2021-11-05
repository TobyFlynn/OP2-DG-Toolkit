#include "dg_utils.h"

// Calculate differentiation matrices
void DGUtils::dMatrices2D(const arma::vec &r, const arma::vec &s,
                          const arma::mat &V, const int N, arma::mat &dr,
                          arma::mat &ds) {
  arma::mat vDr, vDs;
  gradVandermonde2D(r, s, N, vDr, vDs);

  dr.reset();
  ds.reset();

  dr = vDr * arma::inv(V);
  ds = vDs * arma::inv(V);
}
