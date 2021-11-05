#include "dg_constants.h"

#include "dg_utils.h"

void DGConstants::setup(const int n) {
  // Set order
  N = n;
  // Set num points and num face points
  DGUtils::basic_constants(N, &Np, &Nfp);

  // Set the coordinates of the points on the refernece triangle
  arma::vec x, y, r, s;
  DGUtils::setRefXY(N, x, y);
  DGUtils::xy2rs(x, y, r, s);

  // Reference element matrices
  arma::mat V = DGUtils::vandermonde2D(r, s, N);
  arma::mat invV = arma::inv(V);
  arma::mat MassMatrix = invV.t() * invV;
  // TODO
}
