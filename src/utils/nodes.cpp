#include "dg_utils.h"

#include <cmath>

// Uses Warp & Blend to get optimal positions of points on an equilateral
// triangle
void DGUtils::setRefXY(const int N, arma::vec &x, arma::vec &y) {
  // Get basic constants
  int Np, Nfp;
  numNodes2D(N, &Np, &Nfp);

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
    zeroF[i] = fabs(in[i]) < 1.0 - 1e-10 ? 1.0 : 0.0;
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

/*********************
  3D Node Functions
**********************/

// Calculate 1D warp factor
arma::vec warp1D(const int N, const arma::vec &x, const arma::vec &out) {
  arma::vec warp(arma::size(out), arma::fill::zeros);
  arma::vec eq(arma::size(out), arma::fill::zeros);

  for(int i = 0; i < N + 1; i++) {
    eq(i) = -1.0 + 2.0 * (N - i) / (double)N;
  }

  for(int i = 0; i < N + 1; i++) {
    double d = x(i) - eq(i);
    arma::vec dvec;
    bool vecSet = false;
    for(int j = 1; j < N; j++) {
      if(i != j) {
        if(vecSet) {
          dvec = dvec % (out - eq(j)) / (eq(i) - eq(j));
        } else {
          dvec = d * (out - eq(j)) / (eq(i) - eq(j));
          vecSet = true;
        }
      }
    }

    if(!vecSet) {
      dvec = arma::vec(arma::size(out), arma::fill::none);
      dvec.fill(d);
    }

    if(i != 0)
      dvec = -dvec / (eq(i) - eq(0));

    if(i != N)
      dvec = dvec / (eq(i) - eq(N));

    warp = warp + dvec;
  }

  return warp;
}

// Calculate 2D warp factor
void warp2D(const int N, const double p1, const arma::vec &l1,
            const arma::vec &l2, const arma::vec &l3, arma::vec &warpx,
            arma::vec &warpy) {
  arma::vec gaussX = -DGUtils::jacobiGL(0.0, 0.0, N);
  arma::vec blend1 = l2 % l3;
  arma::vec blend2 = l1 % l3;
  arma::vec blend3 = l1 % l2;

  arma::vec warpfactor1 = 4.0 * warp1D(N, gaussX, l3 - l2);
  arma::vec warpfactor2 = 4.0 * warp1D(N, gaussX, l1 - l3);
  arma::vec warpfactor3 = 4.0 * warp1D(N, gaussX, l2 - l1);

  arma::vec warp1 = blend1 % warpfactor1 % (1.0 + arma::square(p1 * l1));
  arma::vec warp2 = blend2 % warpfactor2 % (1.0 + arma::square(p1 * l2));
  arma::vec warp3 = blend3 % warpfactor3 % (1.0 + arma::square(p1 * l3));

  warpx = 1.0 * warp1 + cos(2.0 * DGUtils::PI / 3.0) * warp2 + cos(4.0 * DGUtils::PI / 3.0) * warp3;
  warpy = 0.0 * warp1 + sin(2.0 * DGUtils::PI / 3.0) * warp2 + sin(4.0 * DGUtils::PI / 3.0) * warp3;
}

// Uses Warp & Blend to get optimal positions of points on a reference
// tetrahedron
void DGUtils::setRefXYZ(const int N, arma::vec &x, arma::vec &y,
                        arma::vec &z) {
  // Get basic constants
  int Np, Nfp;
  numNodes3D(N, &Np, &Nfp);

  // Set size of result coord vecs
  x.reset(); y.reset(); z.reset();
  x.set_size(Np); y.set_size(Np); z.set_size(Np);

  // Optimal values of alpha up to N = 16
  double alphaVals[] = {
    0.0, 0.0, 0.0, 0.1002, 1.1332, 1.5608, 1.3413, 1.2577, 1.1603, 1.10153,
    0.6080, 0.4523, 0.8856, 0.8717, 0.9655
  };

  // Set optimal value of alpha for warp & blend
  double alpha = 1.0;
  if(N < 15)
    alpha = alphaVals[N - 1];

  // Equidistance points
  arma::vec r(Np), s(Np), t(Np);
  int ind = 0;
  for(int n = 0; n < N + 1; n++) {
    for(int m = 0; m < N + 1 - n; m++) {
      for(int q = 0; q < N + 1 - n - m; q++) {
        r[ind] = -1.0 + q * 2.0 / (double)N;
        s[ind] = -1.0 + m * 2.0 / (double)N;
        t[ind] = -1.0 + n * 2.0 / (double)N;
        ind++;
      }
    }
  }

  arma::vec l1 = (1.0 + t) / 2.0;
  arma::vec l2 = (1.0 + s) / 2.0;
  arma::vec l3 = -(1.0 + r + s + t) / 2.0;
  arma::vec l4 = (1.0 + r) / 2.0;

  // Vertices of tetrahedron
  arma::rowvec v1 = {-1.0, -1.0 / sqrt(3.0), -1.0 / sqrt(6.0)};
  arma::rowvec v2 = {1.0, -1.0 / sqrt(3.0), -1.0 / sqrt(6.0)};
  arma::rowvec v3 = {0.0, 2.0 / sqrt(3.0), -1.0 / sqrt(6.0)};
  arma::rowvec v4 = {0.0, 0.0, 3.0 / sqrt(6.0)};

  // Orthogonal axis tangents on faces 1-4
  arma::mat t1(4, 3), t2(4, 3);
  t1.row(0) = v2 - v1;
  t1.row(1) = v2 - v1;
  t1.row(2) = v3 - v2;
  t1.row(3) = v3 - v1;
  t2.row(0) = v3 - 0.5 * (v1 + v2);
  t2.row(1) = v4 - 0.5 * (v1 + v2);
  t2.row(2) = v4 - 0.5 * (v2 + v3);
  t2.row(3) = v4 - 0.5 * (v1 + v3);

  // Normalise tangents
  for(int i = 0; i < 4; i++) {
    t1.row(i) = t1.row(i) / arma::norm(t1.row(i));
    t2.row(i) = t2.row(i) / arma::norm(t2.row(i));
  }

  // Warp and blend for each face
  arma::mat xyz = l3 * v1 + l4 * v2 + l2 * v3 + l1 * v4;
  arma::mat shift(arma::size(xyz), arma::fill::zeros);
  for(int i = 0; i < 4; i++) {
    arma::vec la, lb, lc, ld;
    if(i == 0) {
      la = l1; lb = l2; lc = l3; ld = l4;
    } else if(i == 1) {
      la = l2; lb = l1; lc = l3; ld = l4;
    } else if(i == 2) {
      la = l3; lb = l1; lc = l4; ld = l2;
    } else {
      la = l4; lb = l1; lc = l3; ld = l2;
    }

    // Compute warp tangential to face
    arma::vec warp1, warp2;
    warp2D(N, alpha, lb, lc, ld, warp1, warp2);

    arma::vec blend = lb % lc % ld;
    arma::vec denom = (lb + 0.5 * la) % (lc + 0.5 * la) % (ld + 0.5 * la);
    arma::uvec ids = arma::find(denom > 1e-10);
    blend(ids) = (1.0 + arma::square(alpha * la(ids))) % blend(ids) / denom(ids);

    shift = shift + (blend % warp1) * t1.row(i) + (blend % warp2) * t2.row(i);

    ids = find(la < 1e-10 && ((lb > 1e-10) + (lc > 1e-10) + (ld > 1e-10) < 3));
    shift.rows(ids) = warp1(ids) * t1.row(i) + warp2(ids) * t2.row(i);
  }

  xyz = xyz + shift;
  x = xyz.col(0);
  y = xyz.col(1);
  z = xyz.col(2);
}

// Convert from x-y-z coordinates to r-s-t coordinates
void DGUtils::xyz2rst(const arma::vec &x, const arma::vec &y,
                      const arma::vec &z, arma::vec &r, arma::vec &s,
                      arma::vec &t) {
  // Vertices of tetrahedron
  arma::rowvec v1 = {-1.0, -1.0 / sqrt(3.0), -1.0 / sqrt(6.0)};
  arma::rowvec v2 = {1.0, -1.0 / sqrt(3.0), -1.0 / sqrt(6.0)};
  arma::rowvec v3 = {0.0, 2.0 / sqrt(3.0), -1.0 / sqrt(6.0)};
  arma::rowvec v4 = {0.0, 0.0, 3.0 / sqrt(6.0)};

  arma::mat rhs = arma::join_vert(x.t(), y.t(), z.t());
  rhs = rhs - 0.5 * (v2.t() + v3.t() + v4.t() - v1.t()) * arma::ones<arma::mat>(1, x.n_elem);
  arma::mat a = arma::join_horiz(0.5 * (v2 - v1).t(), 0.5 * (v3 - v1).t(), 0.5 * (v4 - v1).t());
  arma::mat rst = arma::solve(a, rhs);
  r = rst.row(0).t();
  s = rst.row(1).t();
  t = rst.row(2).t();
}

// Convert from r-s-t coordinates to a-b-c coordinates
void DGUtils::rst2abc(const arma::vec &r, const arma::vec &s,
                      const arma::vec &t, arma::vec &a, arma::vec &b,
                      arma::vec &c) {
  const int np = r.n_elem;
  a.zeros(np);
  b.zeros(np);
  c.zeros(np);

  for(int i = 0; i < np; i++) {
    if(s(i) + t(i) != 0) {
      a(i) = 2.0 * (1.0 + r(i)) / (-s(i) - t(i)) - 1.0;
    } else {
      a(i) = -1.0;
    }

    if(t(i) != 1.0) {
      b(i) = 2.0 * (1.0 + s(i)) / (1.0 - t(i)) - 1.0;
    } else {
      b(i) = -1.0;
    }
  }

  c = t;
}