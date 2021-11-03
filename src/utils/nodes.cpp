#include "dg_utils.h"

#include <cmath>

// Uses Warp & Blend to get optimal positions of points on the 'model'
// equilateral triangle element
void DGUtils::setXY(std::vector<double> &x, std::vector<double> &y,
                    const int N) {
  // Get basic constants
  int Np, Nfp;
  basic_constants(N, &Np, &Nfp);

  // Optimal values of alpha up to N = 16
  double alphaVals[] = {
    0.0, 0.0, 1.4152, 0.1001, 0.2751, 0.98, 1.0999, 1.2832, 1.3648, 1.4773,
    1.4959, 1.5743, 1.5770, 1.6223, 1.6258
  };

  // Set optimal value of alpha for warp & blend
  double alpha = 5.0 / 3.0;
  if(N < 16)
    alpha = alphaVals[N - 1];

  // Equidistance points on the equilateral triangle
  std::vector<double> l1(Np), l2(Np), l3(Np);
  std::vector<double> blend1(Np), blend2(Np), blend3(Np);
  std::vector<double> warp1Arg(Np), warp2Arg(Np), warp3Arg(Np);
  int ind = 0;
  for(int n = 0; n < N + 1; n++) {
    for(int m = 0; m < N + 1 - n; m++) {
      l1[ind] = (double)n / (double)N;
      l3[ind] = (double)m / (double)N;
      l2[ind] = 1.0 - l1[ind] - l3[ind];
      x[ind] = l3[ind] - l2[ind];
      y[ind] = (2.0 * l1[ind] - l2[ind] - l3[ind]) / sqrt(3.0);
      // Blending functions at each node (for each edge)
      blend1[ind] = 4.0 * l2[ind] * l3[ind];
      blend2[ind] = 4.0 * l1[ind] * l3[ind];
      blend3[ind] = 4.0 * l1[ind] * l2[ind];
      // Arguments needed for calculating amount of warp required
      warp1Arg[ind] = l3[ind] - l2[ind];
      warp2Arg[ind] = l1[ind] - l3[ind];
      warp3Arg[ind] = l2[ind] - l1[ind];
      ind++;
    }
  }

  // Get amount of warp for each node, for each face
  std::vector<double> warpf1 = warpFactor(warp1Arg, N);
  std::vector<double> warpf2 = warpFactor(warp2Arg, N);
  std::vector<double> warpf3 = warpFactor(warp3Arg, N);

  for(int i = 0; i < Np; i++) {
    // Combine warp and blend
    double warp1 = blend1[i] * warpf1[i] * (1.0 + (alpha * l1[ind]) * (alpha * l1[ind]));
    double warp2 = blend2[i] * warpf2[i] * (1.0 + (alpha * l2[ind]) * (alpha * l2[ind]));
    double warp3 = blend3[i] * warpf3[i] * (1.0 + (alpha * l3[ind]) * (alpha * l3[ind]));
    // Apply all deformations to equidistance points
    x[i] += 1.0 * warp1 + cos(2.0 * PI / 3.0) * warp2 + cos(4.0 * PI / 3.0) * warp3;
    y[i] += 0.0 * warp1 + sin(2.0 * PI / 3.0) * warp2 + sin(4.0 * PI / 3.0) * warp3;
  }
}

// Calculate warp function based on in interpolation nodes
std::vector<double> DGUtils::warpFactor(std::vector<double> in, const int N) {
  std::vector<double> lglPts = jacobiGL(0.0, 0.0, N);
  // TODO
}
