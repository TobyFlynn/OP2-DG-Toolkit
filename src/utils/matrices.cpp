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

// Surface to volume lift matrix
arma::mat DGUtils::lift2D(const arma::vec &r, const arma::vec &s,
                          const arma::uvec &fmask, const arma::mat &V,
                          const int N) {
  int Np, Nfp;
  DGUtils::basic_constants(N, &Np, &Nfp);
  arma::mat eMat(Np, 3 * Nfp);

  arma::vec faceR = r(fmask(arma::span(0, Nfp - 1)));
  arma::mat v1D   = DGUtils::vandermonde1D(faceR, N);
  arma::mat mE    = arma::inv(v1D * v1D.t());
  arma::uvec col  = arma::regspace<arma::uvec>(0, Nfp - 1);
  eMat.submat(fmask(arma::span(0, Nfp - 1)), col) = mE;

  faceR = r(fmask(arma::span(Nfp, 2 * Nfp - 1)));
  v1D   = DGUtils::vandermonde1D(faceR, N);
  mE    = arma::inv(v1D * v1D.t());
  col   = arma::regspace<arma::uvec>(Nfp, 2 * Nfp - 1);
  eMat.submat(fmask(arma::span(Nfp, 2 * Nfp - 1)), col) = mE;

  arma::vec faceS = s(fmask(arma::span(2 * Nfp, 3 * Nfp - 1)));
  v1D             = DGUtils::vandermonde1D(faceS, N);
  mE              = arma::inv(v1D * v1D.t());
  col             = arma::regspace<arma::uvec>(2 * Nfp, 3 * Nfp - 1);
  eMat.submat(fmask(arma::span(2 * Nfp, 3 * Nfp - 1)), col) = mE;

  return V * (V.t() * eMat);
}

// Interpolation matrix
arma::mat DGUtils::interpMatrix2D(const arma::vec &r, const arma::vec &s,
                         const arma::mat &invV, const int N) {
  arma::mat newV = vandermonde2D(r, s, N);
  return newV * invV;
}
