#include "dg_constants.h"

#include "dg_utils.h"

void DGConstants::setup(const int n) {
  // Set order
  N = n;
  // Set num points and num face points
  DGUtils::basic_constants(N, &Np, &Nfp);

  // Set the coordinates of the points on the refernece triangle
  DGUtils::setRefXY(N, x_, y_);
  DGUtils::xy2rs(x_, y_, r_, s_);

  // Reference element matrices
  V_ = DGUtils::vandermonde2D(r_, s_, N);
  invV_ = arma::inv(V_);
  MassMatrix_ = invV_.t() * invV_;
  DGUtils::dMatrices2D(r_, s_, V_, N, Dr_, Ds_);

  // FMask
  fmask1_ = arma::find(arma::abs(s_ + 1)  < 1e-12);
  fmask2_ = arma::find(arma::abs(r_ + s_) < 1e-12);
  fmask3_ = arma::find(arma::abs(r_ + 1)  < 1e-12);
  fmask3_ = arma::reverse(fmask3_);
  fmask_  = arma::join_horiz(fmask1_, fmask2_, fmask3_);

  // LIFT matrix
  lift_ = DGUtils::lift2D(r_, s_, fmask_, V_, N);

  // Weak operators
  arma::mat Vr, Vs;
  DGUtils::gradVandermonde2D(r_, s_, N, Vr, Vs);
  Drw_ = (V_ * Vr.t()) * arma::inv(V_ * V_.t());
  Dsw_ = (V_ * Vs.t()) * arma::inv(V_ * V_.t());
}

void DGConstants::gauss(const int nGauss) {
  arma::vec g_x, g_w;
  DGUtils::jacobiGQ(0.0, 0.0, nGauss - 1, g_x, g_w);

  arma::vec face1r = g_x;
  arma::vec face2r = -g_x;
  arma::vec face3r = -arma::ones<arma::vec>(nGauss);
  arma::vec face1s = -arma::ones<arma::vec>(nGauss);
  arma::vec face2s = g_x;
  arma::vec face3s = -g_x;

  arma::mat interp1 = DGUtils::vandermonde2D(face1r, face1s, N) * invV_;
  arma::mat interp2 = DGUtils::vandermonde2D(face2r, face2s, N) * invV_;
  arma::mat interp3 = DGUtils::vandermonde2D(face3r, face3s, N) * invV_;

  arma::mat interp = arma::join_vert(interp1, interp2, interp3);
}
