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

// Calculate differentiation matrices
void DGUtils::dMatrices3D(const arma::vec &r, const arma::vec &s,
                          const arma::vec &t, const arma::mat &V, const int N,
                          arma::mat &dr, arma::mat &ds, arma::mat &dt) {
  arma::mat vDr, vDs, vDt;
  gradVandermonde3D(r, s, t, N, vDr, vDs, vDt);

  dr = vDr * arma::inv(V);
  ds = vDs * arma::inv(V);
  dt = vDt * arma::inv(V);
}

// Surface to volume lift matrix
arma::mat DGUtils::lift2D(const arma::vec &r, const arma::vec &s,
                          const arma::uvec &fmask, const arma::mat &V,
                          const int N) {
  int Np, Nfp;
  DGUtils::numNodes2D(N, &Np, &Nfp);
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

// Calculate 3D surface to volume lift operator
arma::mat DGUtils::eMat3D(const arma::vec &r, const arma::vec &s,
                          const arma::vec &t, const arma::uvec &fmask,
                          const int N) {
  int Np, Nfp;
  numNodes3D(N, &Np, &Nfp);

  arma::mat eMat(Np, 4 * Nfp, arma::fill::zeros);

  arma::vec faceR = r(fmask(arma::span(0, Nfp - 1)));
  arma::vec faceS = s(fmask(arma::span(0, Nfp - 1)));
  arma::mat vFace = vandermonde2D(faceR, faceS, N);
  arma::mat mFace = arma::inv(vFace * vFace.t());
  arma::uvec col  = arma::regspace<arma::uvec>(0, Nfp - 1);
  eMat.submat(fmask(arma::span(0, Nfp - 1)), col) += mFace;

  faceR = r(fmask(arma::span(Nfp, 2 * Nfp - 1)));
  faceS = t(fmask(arma::span(Nfp, 2 * Nfp - 1)));
  vFace = vandermonde2D(faceR, faceS, N);
  mFace = arma::inv(vFace * vFace.t());
  col   = arma::regspace<arma::uvec>(Nfp, 2 * Nfp - 1);
  eMat.submat(fmask(arma::span(Nfp, 2 * Nfp - 1)), col) += mFace;

  faceR = s(fmask(arma::span(2 * Nfp, 3 * Nfp - 1)));
  faceS = t(fmask(arma::span(2 * Nfp, 3 * Nfp - 1)));
  vFace = vandermonde2D(faceR, faceS, N);
  mFace = arma::inv(vFace * vFace.t());
  col   = arma::regspace<arma::uvec>(2 * Nfp, 3 * Nfp - 1);
  eMat.submat(fmask(arma::span(2 * Nfp, 3 * Nfp - 1)), col) += mFace;

  faceR = s(fmask(arma::span(3 * Nfp, 4 * Nfp - 1)));
  faceS = t(fmask(arma::span(3 * Nfp, 4 * Nfp - 1)));
  vFace = vandermonde2D(faceR, faceS, N);
  mFace = arma::inv(vFace * vFace.t());
  col   = arma::regspace<arma::uvec>(3 * Nfp, 4 * Nfp - 1);
  eMat.submat(fmask(arma::span(3 * Nfp, 4 * Nfp - 1)), col) += mFace;

  return eMat;
}

// Calculate 3D surface to volume lift operator
arma::mat DGUtils::lift3D(const arma::vec &r, const arma::vec &s,
                          const arma::vec &t, const arma::uvec &fmask,
                          const arma::mat &v, const int N) {
  return v * (v.t() * eMat3D(r, s, t, fmask, N));
}

// Interpolation matrix
arma::mat DGUtils::interpMatrix2D(const arma::vec &r, const arma::vec &s,
                                  const arma::mat &invV, const int N) {
  arma::mat newV = vandermonde2D(r, s, N);
  return newV * invV;
}

arma::mat DGUtils::interpMatrix3D(const arma::vec &r, const arma::vec &s,
                                  const arma::vec &t, const arma::mat &invV,
                                  const int N) {
  arma::mat v = vandermonde3D(r, s, t, N);
  return v * invV;
}

// Calculate the mass matrix of each face
void DGUtils::faceMassMatrix3D(const arma::vec &r, const arma::vec &s,
                               const arma::vec &t, const arma::uvec &fmask,
                               const arma::mat &v, const int N,
                               arma::mat &face0, arma::mat &face1,
                               arma::mat &face2, arma::mat &face3) {
  int Np, Nfp;
  numNodes3D(N, &Np, &Nfp);

  arma::mat zMat(Np, Np, arma::fill::zeros);
  face0 = zMat;
  face1 = zMat;
  face2 = zMat;
  face3 = zMat;

  arma::vec faceR = r(fmask(arma::span(0, Nfp - 1)));
  arma::vec faceS = s(fmask(arma::span(0, Nfp - 1)));
  arma::mat vFace = vandermonde2D(faceR, faceS, N);
  face0.submat(fmask(arma::span(0, Nfp - 1)), fmask(arma::span(0, Nfp - 1))) += arma::inv(vFace * vFace.t());

  faceR = r(fmask(arma::span(Nfp, 2 * Nfp - 1)));
  faceS = t(fmask(arma::span(Nfp, 2 * Nfp - 1)));
  vFace = vandermonde2D(faceR, faceS, N);
  face1.submat(fmask(arma::span(Nfp, 2 * Nfp - 1)), fmask(arma::span(Nfp, 2 * Nfp - 1))) += arma::inv(vFace * vFace.t());

  faceR = s(fmask(arma::span(2 * Nfp, 3 * Nfp - 1)));
  faceS = t(fmask(arma::span(2 * Nfp, 3 * Nfp - 1)));
  vFace = vandermonde2D(faceR, faceS, N);
  face2.submat(fmask(arma::span(2 * Nfp, 3 * Nfp - 1)), fmask(arma::span(2 * Nfp, 3 * Nfp - 1))) += arma::inv(vFace * vFace.t());

  faceR = s(fmask(arma::span(3 * Nfp, 4 * Nfp - 1)));
  faceS = t(fmask(arma::span(3 * Nfp, 4 * Nfp - 1)));
  vFace = vandermonde2D(faceR, faceS, N);
  face3.submat(fmask(arma::span(3 * Nfp, 4 * Nfp - 1)), fmask(arma::span(3 * Nfp, 4 * Nfp - 1))) += arma::inv(vFace * vFace.t());
}

arma::mat DGUtils::filterMatrix3D(const arma::mat &v, const arma::mat &invV,
                                  const int N, const int Nc, const int s,
                                  const DG_FP alpha) {
  int diag_ind = 0;
  arma::mat tmp(v.n_rows, v.n_rows, arma::fill::zeros);
  for(int i = 0; i < N + 1; i++) {
    for(int j = 0; j < N - i + 1; j++) {
      for(int k = 0; k < N - i - j + 1; k++) {
        if(i + j + k >= Nc) {
          tmp(diag_ind, diag_ind) = exp(-alpha * pow(((i + j + k - Nc + 1)/(N - Nc + 1)),s));
        } else {
          tmp(diag_ind, diag_ind) = 1.0;
        }
        diag_ind++;
      }
    }
  }
  return v * tmp * invV;
}
