#include "dg_constants.h"

#include <cmath>

void DGConstants::setup(const int n) {
  // Set order
  N = n;
  // Number of points per face of triangluar element
  Nfp = N + 1;
  // Number of pointer per element
  Np = (N + 1) * (N + 2) / 2;

  // Set the coordinates of the points on the 'model' equilateral triangle
  std::vector<double> x(Np);
  std::vector<double> y(Np);
  setXY(x, y);
}
