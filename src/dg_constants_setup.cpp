#include "dg_constants.h"

#include "dg_utils.h"

#include <cstring>

#include "dg_compiler_defs.h"

double r_g[DG_ORDER * DG_NP];
double s_g[DG_ORDER * DG_NP];
double ones_g[DG_ORDER * DG_NP];
double v_g[DG_ORDER * DG_NP * DG_NP];
double invV_g[DG_ORDER * DG_NP * DG_NP];
double mass_g[DG_ORDER * DG_NP * DG_NP];
double invMass_g[DG_ORDER * DG_NP * DG_NP];
double Dr_g[DG_ORDER * DG_NP * DG_NP];
double Ds_g[DG_ORDER * DG_NP * DG_NP];
double Drw_g[DG_ORDER * DG_NP * DG_NP];
double Dsw_g[DG_ORDER * DG_NP * DG_NP];
double lift_g[DG_ORDER * DG_NP * 3 * DG_NPF];

int FMASK[DG_ORDER * 3 * DG_NPF];
int DG_CONSTANTS[DG_ORDER * 5];

double cubW_g[DG_ORDER * DG_CUB_NP];
double cubV_g[DG_ORDER * DG_CUB_NP * DG_NP];
double cubDr_g[DG_ORDER * DG_CUB_NP * DG_NP];
double cubDs_g[DG_ORDER * DG_CUB_NP * DG_NP];
double cubVDr_g[DG_ORDER * DG_CUB_NP * DG_NP];
double cubVDs_g[DG_ORDER * DG_CUB_NP * DG_NP];

double gaussW_g[DG_ORDER * DG_GF_NP];
double gInterp_g[DG_ORDER * DG_G_NP * DG_NP];
double gFInterp0_g[DG_ORDER * DG_GF_NP * DG_NP];
double gFInterp1_g[DG_ORDER * DG_GF_NP * DG_NP];
double gFInterp2_g[DG_ORDER * DG_GF_NP * DG_NP];
double gF0Dr_g[DG_ORDER * DG_GF_NP * DG_NP];
double gF0Ds_g[DG_ORDER * DG_GF_NP * DG_NP];
double gF1Dr_g[DG_ORDER * DG_GF_NP * DG_NP];
double gF1Ds_g[DG_ORDER * DG_GF_NP * DG_NP];
double gF2Dr_g[DG_ORDER * DG_GF_NP * DG_NP];
double gF2Ds_g[DG_ORDER * DG_GF_NP * DG_NP];

double invMass_gInterpT_g[DG_ORDER * DG_G_NP * DG_NP];

// Effectively a 2D array of interp matrices. Size [DG_ORDER][DG_ORDER]
// To get an interp array:
// int ind = ((order_from - 1) * DG_ORDER + (order_to - 1)) * DG_NP * DG_NP
double order_interp_g[DG_ORDER * DG_ORDER * DG_NP * DG_NP];

extern DGConstants *constants[DG_ORDER + 1];

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
  fmask_  = arma::join_cols(fmask1_, fmask2_, fmask3_);

  // LIFT matrix
  lift_ = DGUtils::lift2D(r_, s_, fmask_, V_, N);

  // Weak operators
  arma::mat Vr, Vs;
  DGUtils::gradVandermonde2D(r_, s_, N, Vr, Vs);
  Drw_ = (V_ * Vr.t()) * arma::inv(V_ * V_.t());
  Dsw_ = (V_ * Vs.t()) * arma::inv(V_ * V_.t());

  int intN = ceil(3.0 * N / 2.0);
  cubature(2 * (intN + 1));
  // gauss(intN + 1);
  // Use same Gauss points for every order (makes p-adaptivity much more simple)
  intN = ceil(3.0 * DG_ORDER / 2.0);
  gauss(intN + 1);

  arma::vec ones = arma::ones<arma::vec>(r_.n_elem);
  arma::mat invMass = arma::inv(MassMatrix_);
  arma::mat cub_V_Dr = cub_V_ * Dr_;
  arma::mat cub_V_Ds = cub_V_ * Ds_;
  arma::mat gauss_i1_Dr = gauss_interp1_ * Dr_;
  arma::mat gauss_i1_Ds = gauss_interp1_ * Ds_;
  arma::mat gauss_i2_Dr = gauss_interp2_ * Dr_;
  arma::mat gauss_i2_Ds = gauss_interp2_ * Ds_;
  arma::mat gauss_i3_Dr = gauss_interp3_ * Dr_;
  arma::mat gauss_i3_Ds = gauss_interp3_ * Ds_;
  // arma::mat invMass_gauss_interpT = invMass * gauss_interp_.t();
  arma::mat invMass_gauss_interpT = V_ * V_.t() * gauss_interp_.t();

  // Copy armadillo vecs and mats to global memory

  memcpy(&r_g[(N - 1) * DG_NP], r_.memptr(), r_.n_elem * sizeof(double));
  memcpy(&s_g[(N - 1) * DG_NP], s_.memptr(), s_.n_elem * sizeof(double));
  memcpy(&ones_g[(N - 1) * DG_NP], ones.memptr(), ones.n_elem * sizeof(double));
  memcpy(&v_g[(N - 1) * DG_NP * DG_NP], V_.memptr(), V_.n_elem * sizeof(double));
  memcpy(&invV_g[(N - 1) * DG_NP * DG_NP], invV_.memptr(), invV_.n_elem * sizeof(double));
  memcpy(&mass_g[(N - 1) * DG_NP * DG_NP], MassMatrix_.memptr(), MassMatrix_.n_elem * sizeof(double));
  memcpy(&invMass_g[(N - 1) * DG_NP * DG_NP], invMass.memptr(), invMass.n_elem * sizeof(double));
  memcpy(&Dr_g[(N - 1) * DG_NP * DG_NP], Dr_.memptr(), Dr_.n_elem * sizeof(double));
  memcpy(&Ds_g[(N - 1) * DG_NP * DG_NP], Ds_.memptr(), Ds_.n_elem * sizeof(double));
  memcpy(&Drw_g[(N - 1) * DG_NP * DG_NP], Drw_.memptr(), Drw_.n_elem * sizeof(double));
  memcpy(&Dsw_g[(N - 1) * DG_NP * DG_NP], Dsw_.memptr(), Dsw_.n_elem * sizeof(double));
  memcpy(&lift_g[(N - 1) * DG_NP * 3 * DG_NPF], lift_.memptr(), lift_.n_elem * sizeof(double));

  std::vector<int> fmask_int = arma::conv_to<std::vector<int>>::from(fmask_);
  memcpy(&FMASK[(N - 1) * 3 * DG_NPF], fmask_int.data(), fmask_int.size() * sizeof(int));
  // Number of points
  DG_CONSTANTS[(N - 1) * 5 + 0] = Np;
  // Number of face points
  DG_CONSTANTS[(N - 1) * 5 + 1] = Nfp;
  // Number of cubature points
  cNp = cub_w_.n_elem;
  DG_CONSTANTS[(N - 1) * 5 + 2] = cNp;
  // Number of gauss points
  gNp = gauss_w_.n_elem * 3;
  DG_CONSTANTS[(N - 1) * 5 + 3] = gNp;
  // Number of gauss points per edge
  gNfp = gauss_w_.n_elem;
  DG_CONSTANTS[(N - 1) * 5 + 4] = gNfp;

  memcpy(&cubW_g[(N - 1) * DG_CUB_NP], cub_w_.memptr(), cub_w_.n_elem * sizeof(double));
  memcpy(&cubV_g[(N - 1) * DG_CUB_NP * DG_NP], cub_V_.memptr(), cub_V_.n_elem * sizeof(double));
  memcpy(&cubDr_g[(N - 1) * DG_CUB_NP * DG_NP], cub_Dr_.memptr(), cub_Dr_.n_elem * sizeof(double));
  memcpy(&cubDs_g[(N - 1) * DG_CUB_NP * DG_NP], cub_Ds_.memptr(), cub_Ds_.n_elem * sizeof(double));
  memcpy(&cubVDr_g[(N - 1) * DG_CUB_NP * DG_NP], cub_V_Dr.memptr(), cub_V_Dr.n_elem * sizeof(double));
  memcpy(&cubVDs_g[(N - 1) * DG_CUB_NP * DG_NP], cub_V_Ds.memptr(), cub_V_Ds.n_elem * sizeof(double));

  memcpy(&gaussW_g[(N - 1) * DG_GF_NP], gauss_w_.memptr(), gauss_w_.n_elem * sizeof(double));
  memcpy(&gInterp_g[(N - 1) * DG_G_NP * DG_NP], gauss_interp_.memptr(), gauss_interp_.n_elem * sizeof(double));
  memcpy(&gFInterp0_g[(N - 1) * DG_GF_NP * DG_NP], gauss_interp1_.memptr(), gauss_interp1_.n_elem * sizeof(double));
  memcpy(&gFInterp1_g[(N - 1) * DG_GF_NP * DG_NP], gauss_interp2_.memptr(), gauss_interp2_.n_elem * sizeof(double));
  memcpy(&gFInterp2_g[(N - 1) * DG_GF_NP * DG_NP], gauss_interp3_.memptr(), gauss_interp3_.n_elem * sizeof(double));
  memcpy(&gF0Dr_g[(N - 1) * DG_GF_NP * DG_NP], gauss_i1_Dr.memptr(), gauss_i1_Dr.n_elem * sizeof(double));
  memcpy(&gF0Ds_g[(N - 1) * DG_GF_NP * DG_NP], gauss_i1_Ds.memptr(), gauss_i1_Ds.n_elem * sizeof(double));
  memcpy(&gF1Dr_g[(N - 1) * DG_GF_NP * DG_NP], gauss_i2_Dr.memptr(), gauss_i2_Dr.n_elem * sizeof(double));
  memcpy(&gF1Ds_g[(N - 1) * DG_GF_NP * DG_NP], gauss_i2_Ds.memptr(), gauss_i2_Ds.n_elem * sizeof(double));
  memcpy(&gF2Dr_g[(N - 1) * DG_GF_NP * DG_NP], gauss_i3_Dr.memptr(), gauss_i3_Dr.n_elem * sizeof(double));
  memcpy(&gF2Ds_g[(N - 1) * DG_GF_NP * DG_NP], gauss_i3_Ds.memptr(), gauss_i3_Ds.n_elem * sizeof(double));

  memcpy(&invMass_gInterpT_g[(N - 1) * DG_G_NP * DG_NP], invMass_gauss_interpT.memptr(), invMass_gauss_interpT.n_elem * sizeof(double));
}

void DGConstants::cubature(const int nCub) {
  arma::vec c_r, c_s;
  DGUtils::cubature2D(nCub, c_r, c_s, cub_w_);

  cub_V_ = DGUtils::interpMatrix2D(c_r, c_s, invV_, N);

  DGUtils::dMatrices2D(c_r, c_s, V_, N, cub_Dr_, cub_Ds_);
}

void DGConstants::gauss(const int nGauss) {
  arma::vec g_x;
  DGUtils::jacobiGQ(0.0, 0.0, nGauss - 1, g_x, gauss_w_);

  arma::vec face1r = g_x;
  arma::vec face2r = -g_x;
  arma::vec face3r = -arma::ones<arma::vec>(nGauss);
  arma::vec face1s = -arma::ones<arma::vec>(nGauss);
  arma::vec face2s = g_x;
  arma::vec face3s = -g_x;

  gauss_interp1_ = DGUtils::vandermonde2D(face1r, face1s, N) * invV_;
  gauss_interp2_ = DGUtils::vandermonde2D(face2r, face2s, N) * invV_;
  gauss_interp3_ = DGUtils::vandermonde2D(face3r, face3s, N) * invV_;

  gauss_interp_ = arma::join_vert(gauss_interp1_, gauss_interp2_, gauss_interp3_);
}

void DGConstants::calc_interp_mats() {
  for(int n = 1; n <= DG_ORDER; n++) {
    if(n != N) {
      if(n > N) {
        interp_[n - 1] = DGUtils::interpMatrix2D(constants[n]->r_, constants[n]->s_, invV_, N);
      } else {
        interp_[n - 1] = DGUtils::interpMatrix2D(r_, s_, constants[n]->invV_, constants[n]->N).t();
      }
      // arma::mat newV = DGUtils::vandermonde2D(constants[n]->r_, constants[n]->s_, N);
      // interp_[n - 1] = newV * invV_;
      // interp_[n - 1] = arma::inv(constants[n]->MassMatrix_) * (V_ * newV.t());

      // interp_[n - 1] = DGUtils::interpMatrix2D(constants[n]->r_, constants[n]->s_, invV_, N);
      memcpy(&order_interp_g[((N - 1) * DG_ORDER + (n - 1)) * DG_NP * DG_NP], interp_[n - 1].memptr(), interp_[n - 1].n_elem * sizeof(double));
    }
  }
}
