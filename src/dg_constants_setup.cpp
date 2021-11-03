#include "dg_constants.h"

#include "dg_utils.h"

void DGConstants::setup(const int n) {
  // Set order
  N = n;
  // Set num points and num face points
  DGUtils::basic_constants(N, &Np, &Nfp);

  // Set the coordinates of the points on the 'model' equilateral triangle
  std::vector<double> x(Np);
  std::vector<double> y(Np);
  DGUtils::setXY(x, y, N);
}
