#include "dg_utils.h"

#include <cmath>

// Uses Warp & Blend to get optimal positions of points on an equilateral
// triangle
void DGUtils::setRefXY(const int N, arma::vec &x, arma::vec &y) {
  // Get basic constants
  int Np, Nfp;
  basic_constants(N, &Np, &Nfp);

  // Set size of result coord vecs
  x.reset(); y.reset();
  x.set_size(Np); y.set_size(Np);

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
  arma::vec l1(Np), l2(Np), l3(Np);
  arma::vec blend1(Np), blend2(Np), blend3(Np);
  int ind = 0;
  for(int n = 0; n < N + 1; n++) {
    for(int m = 0; m < N + 1 - n; m++) {
      l1[ind] = (double)n / (double)N;
      l3[ind] = (double)m / (double)N;
      ind++;
    }
  }

  l2 = 1.0 - l1 - l3;
  x  = l3 - l2;
  y  = (2.0 * l1 - l2 - l3) / sqrt(3.0);

  // Blending functions at each node (for each edge)
  blend1 = 4.0 * l2 % l3;
  blend2 = 4.0 * l1 % l3;
  blend3 = 4.0 * l1 % l2;

  // Get amount of warp for each node, for each face
  arma::vec warpf1 = warpFactor(l3 - l2, N);
  arma::vec warpf2 = warpFactor(l1 - l3, N);
  arma::vec warpf3 = warpFactor(l2 - l1, N);

  arma::vec warp1 = blend1 % warpf1 % (1.0 + alpha * alpha * l1 % l1);
  arma::vec warp2 = blend2 % warpf2 % (1.0 + alpha * alpha * l2 % l2);
  arma::vec warp3 = blend3 % warpf3 % (1.0 + alpha * alpha * l3 % l3);

  x = x + warp1 + cos(2.0 * PI / 3.0) * warp2 + cos(4.0 * PI / 3.0) * warp3;
  y = y + sin(2.0 * PI / 3.0) * warp2 + sin(4.0 * PI / 3.0) * warp3;
}

// Calculate warp function based on in interpolation nodes
arma::vec DGUtils::warpFactor(const arma::vec &in, const int N) {
  arma::vec lglPts = jacobiGL(0.0, 0.0, N);
  arma::vec rEq    = arma::linspace(-1.0, 1.0, N + 1);
  arma::mat v1D    = vandermonde1D(rEq, N);
  arma::mat pMat(N + 1, in.n_elem);
  for(int i = 0; i < N + 1; i++) {
    pMat.row(i) = jacobiP(in, 0.0, 0.0, i).t();
  }
  arma::mat lMat = arma::solve(v1D.t(), pMat);

  arma::vec warp = lMat.t() * (lglPts - rEq);

  arma::vec zeroF(in.n_elem);
  arma::vec sF(in.n_elem);
  for(int i = 0; i < in.n_elem; i++) {
    zeroF[i] = abs(in[i]) < 1.0 - 1e-10 ? 1.0 : 0.0;
    sF[i]    = 1.0 - (zeroF[i] * in[i]) * (zeroF[i] * in[i]);
  }

  return warp / sF + warp % (zeroF - 1.0);
}

// Convert from x-y coordinates in equilateral triangle to r-s coordinates of
// the reference triagnle
void DGUtils::xy2rs(const arma::vec &x, const arma::vec &y, arma::vec &r,
                    arma::vec &s) {
  arma::vec l1 = (sqrt(3.0) * y + 1.0) / 3.0;
  arma::vec l2 = (-3.0 * x - sqrt(3.0) * y + 2.0) / 6.0;
  arma::vec l3 = (3.0 * x - sqrt(3.0) * y + 2.0) / 6.0;

  r.set_size(arma::size(x));
  s.set_size(arma::size(x));
  r = l3 - l2 - l1;
  s = l1 - l2 - l3;
}

// Convert from r-s coordinates to a-b coordinates
void DGUtils::rs2ab(const arma::vec &r, const arma::vec &s, arma::vec &a,
                    arma::vec &b) {
  a.set_size(arma::size(r));
  b.set_size(arma::size(r));
  for(int i = 0; i < r.n_elem; i++) {
    if(s[i] != 1.0) {
      a[i] = 2.0 * (1.0 + r[i]) / (1.0 - s[i]) - 1.0;
    } else {
      a[i] = -1.0;
    }
  }
  b = s;
}
