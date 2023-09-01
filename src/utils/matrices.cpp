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
arma::mat DGUtils::eMat2D(const arma::vec &r, const arma::vec &s,
                          const arma::uvec &fmask, const int N) {
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

  return eMat;
}

arma::mat DGUtils::lift2D(const arma::vec &r, const arma::vec &s,
                          const arma::uvec &fmask, const arma::mat &V,
                          const int N) {
  arma::mat eMat = eMat2D(r, s, fmask, N);

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
void DGUtils::faceMassMatrix2D(const arma::vec &r, const arma::vec &s,
                               const arma::uvec &fmask, const arma::mat &v,
                               const int N, arma::mat &face0, arma::mat &face1,
                               arma::mat &face2) {
  int Np, Nfp;
  numNodes2D(N, &Np, &Nfp);

  arma::mat zMat(Np, Np, arma::fill::zeros);
  face0 = zMat;
  face1 = zMat;
  face2 = zMat;

  arma::vec faceR = r(fmask(arma::span(0, Nfp - 1)));
  arma::mat v1D   = DGUtils::vandermonde1D(faceR, N);
  arma::mat mE    = arma::inv(v1D * v1D.t());
  face0.submat(fmask(arma::span(0, Nfp - 1)), fmask(arma::span(0, Nfp - 1))) += mE;

  faceR = r(fmask(arma::span(Nfp, 2 * Nfp - 1)));
  v1D   = DGUtils::vandermonde1D(faceR, N);
  mE    = arma::inv(v1D * v1D.t());
  face1.submat(fmask(arma::span(Nfp, 2 * Nfp - 1)), fmask(arma::span(Nfp, 2 * Nfp - 1))) += mE;

  arma::vec faceS = s(fmask(arma::span(2 * Nfp, 3 * Nfp - 1)));
  v1D             = DGUtils::vandermonde1D(faceS, N);
  mE              = arma::inv(v1D * v1D.t());
  face2.submat(fmask(arma::span(2 * Nfp, 3 * Nfp - 1)), fmask(arma::span(2 * Nfp, 3 * Nfp - 1))) += mE;
}

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

arma::mat DGUtils::cubaturePMat2D(const arma::vec &r, const arma::vec &s, 
                                  const arma::vec &cubr, const arma::vec &cubs,
                                  const int N) {
  arma::mat V = vandermonde2D(r, s, N);
  arma::mat cubV = vandermonde2D(cubr, cubs, N);

  return V * cubV.t();
}

void DGUtils::cubaturePDwMat2D(const arma::vec &r, const arma::vec &s,
                    const arma::vec &cubr, const arma::vec &cubs,
                    const int N, arma::mat &cubDrw, arma::mat &cubDsw) {
  arma::mat V = vandermonde2D(r, s, N);
  arma::mat cubVr, cubVs;
  gradVandermonde2D(cubr, cubs, N, cubVr, cubVs);

  cubDrw = V * cubVr.t();
  cubDsw = V * cubVs.t();
}

arma::mat DGUtils::cubaturePMat3D(const arma::vec &r, const arma::vec &s, 
                                  const arma::vec &t, const arma::vec &cubr, 
                                  const arma::vec &cubs, const arma::vec &cubt, 
                                  const int N) {
  arma::mat V = vandermonde3D(r, s, t, N);
  arma::mat cubV = vandermonde3D(cubr, cubs, cubt, N);

  return V * cubV.t();
}

void DGUtils::cubaturePDwMat3D(const arma::vec &r, const arma::vec &s, const arma::vec &t, 
                    const arma::vec &cubr, const arma::vec &cubs, const arma::vec &cubt, 
                    const int N, arma::mat &cubDrw, arma::mat &cubDsw, arma::mat &cubDtw) {
  arma::mat V = vandermonde3D(r, s, t, N);
  arma::mat cubVr, cubVs, cubVt;
  gradVandermonde3D(cubr, cubs, cubt, N, cubVr, cubVs, cubVt);

  cubDrw = V * cubVr.t();
  cubDsw = V * cubVs.t();
  cubDtw = V * cubVt.t();
}

void DGUtils::cubatureSurface3d(const arma::vec &r, const arma::vec &s, const arma::vec &t, 
                         const arma::uvec &fmask, const arma::vec &cubr, const arma::vec &cubs,
                         const arma::vec &cubw, const int N, arma::mat &interp, arma::mat &lift) {
  const int npf_cub = cubr.n_elem; 
  int np, npf;
  numNodes3D(N, &np, &npf);
/*
  // interp will be sparse and inefficient - TODO do this differently
  interp.zeros(npf_cub * 4, npf * 4);

  arma::vec faceR = r(fmask(arma::span(0, npf - 1)));
  arma::vec faceS = s(fmask(arma::span(0, npf - 1)));
  arma::mat invVFace = arma::inv(vandermonde2D(faceR, faceS, N));
  interp.submat(arma::span(0, npf_cub - 1), arma::span(0, npf - 1)) = interpMatrix2D(cubr, cubs, invVFace, N);

  faceR = r(fmask(arma::span(npf, 2 * npf - 1)));
  faceS = t(fmask(arma::span(npf, 2 * npf - 1)));
  invVFace = arma::inv(vandermonde2D(faceR, faceS, N));
  interp.submat(arma::span(npf_cub, 2 * npf_cub - 1), arma::span(npf, 2 * npf - 1)) = interpMatrix2D(cubr, cubs, invVFace, N);

  faceR = s(fmask(arma::span(2 * npf, 3 * npf - 1)));
  faceS = t(fmask(arma::span(2 * npf, 3 * npf - 1)));
  invVFace = arma::inv(vandermonde2D(faceR, faceS, N));
  interp.submat(arma::span(2 * npf_cub, 3 * npf_cub - 1), arma::span(2 * npf, 3 * npf - 1)) = interpMatrix2D(cubr, cubs, invVFace, N);

  faceR = s(fmask(arma::span(3 * npf, 4 * npf - 1)));
  faceS = t(fmask(arma::span(3 * npf, 4 * npf - 1)));
  invVFace = arma::inv(vandermonde2D(faceR, faceS, N));
  interp.submat(arma::span(3 * npf_cub, 4 * npf_cub - 1), arma::span(3 * npf, 4 * npf - 1)) = interpMatrix2D(cubr, cubs, invVFace, N);
*/
  arma::vec ir(npf_cub * 4);
  arma::vec is(npf_cub * 4);
  arma::vec it(npf_cub * 4);
  arma::vec iw(npf_cub * 4);

  for(int i = 0; i < npf_cub; i++) {
    ir(i)               = cubr(i);
    ir(i + npf_cub)     = cubr(i);
    ir(i + 2 * npf_cub) = cubr(i);
    ir(i + 3 * npf_cub) = -1.0;

    is(i)               = cubs(i);
    is(i + npf_cub)     = -1.0;
    is(i + 2 * npf_cub) = cubs(i);
    is(i + 3 * npf_cub) = cubr(i);

    it(i)               = -1.0;
    it(i + npf_cub)     = cubs(i);
    it(i + 2 * npf_cub) = -(1.0 + cubr(i) + cubs(i));
    it(i + 3 * npf_cub) = cubs(i);

    iw(i)               = cubw(i);
    iw(i + npf_cub)     = cubw(i);
    iw(i + 2 * npf_cub) = cubw(i);
    iw(i + 3 * npf_cub) = cubw(i);
  }

  arma::mat V = vandermonde3D(r, s, t, N);
  arma::mat tmp_interp = DGUtils::interpMatrix3D(ir, is, it, arma::inv(V), N);

  // interp will be sparse and inefficient - TODO do this differently
  interp.zeros(npf_cub * 4, npf * 4);
  for(int face = 0; face < 4; face++) {
    for(int cub_pt = 0; cub_pt < npf_cub; cub_pt++) {
      for(int nodal_pt = 0; nodal_pt < npf; nodal_pt++) {
        interp(face * npf_cub + cub_pt, face * npf + nodal_pt) = tmp_interp(face * npf_cub + cub_pt, fmask(face * npf + nodal_pt));
      }
    }
  }

  arma::mat diagW(npf_cub * 4, npf_cub * 4);
  diagW.zeros();
  for(int i = 0; i < npf_cub * 4; i++) {
    diagW(i,i) = iw(i);
  }
  lift = V * V.t() * tmp_interp.t() * diagW;
}