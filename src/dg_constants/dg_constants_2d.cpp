#include "dg_constants/dg_constants_2d.h"

#include <stdexcept>

#include "dg_compiler_defs.h"
#include "dg_utils.h"

int FMASK[DG_ORDER * DG_NUM_FACES * DG_NPF];
int DG_CONSTANTS[DG_ORDER * 5];
DG_FP cubW_g[DG_ORDER * DG_CUB_NP];
DG_FP gaussW_g[DG_ORDER * DG_GF_NP];

void save_mat(DG_FP *mem_ptr, arma::mat &mat, const int N, const int max_size) {
  arma::Mat<DG_FP> mat_2 = arma::conv_to<arma::Mat<DG_FP>>::from(mat);
  memcpy(&mem_ptr[(N - 1) * max_size], mat_2.memptr(), mat_2.n_elem * sizeof(DG_FP));
}

void save_vec(DG_FP *mem_ptr, arma::vec &vec, const int N, const int max_size) {
  arma::Col<DG_FP> vec_2 = arma::conv_to<arma::Col<DG_FP>>::from(vec);
  memcpy(&mem_ptr[(N - 1) * max_size], vec_2.memptr(), vec_2.n_elem * sizeof(DG_FP));
}

DGConstants2D::DGConstants2D(const int n_) {
  // Set max order
  N_max = n_;
  // Set max num points and max num face points
  DGUtils::numNodes2D(N_max, &Np_max, &Nfp_max);
  // TODO Set max num of cubature points
  cNp_max = DG_CUB_NP;
  // Set max num of Gauss points
  gNfp_max = ceil(3.0 * DG_ORDER / 2.0) + 1;
  gNp_max = 3 * gNfp_max;

  // Allocate memory for matrices of all orders
  r_ptr = (DG_FP *)calloc(N_max * Np_max, sizeof(DG_FP));
  s_ptr = (DG_FP *)calloc(N_max * Np_max, sizeof(DG_FP));
  v_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  invV_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  mass_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  invMass_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  Dr_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  Ds_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  Drw_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  Dsw_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  lift_ptr = (DG_FP *)calloc(N_max * Np_max * DG_NUM_FACES * Nfp_max, sizeof(DG_FP));
  cubV_ptr = (DG_FP *)calloc(N_max * cNp_max * Np_max, sizeof(DG_FP));
  cubDr_ptr = (DG_FP *)calloc(N_max * cNp_max * Np_max, sizeof(DG_FP));
  cubDs_ptr = (DG_FP *)calloc(N_max * cNp_max * Np_max, sizeof(DG_FP));
  cubVDr_ptr = (DG_FP *)calloc(N_max * cNp_max * Np_max, sizeof(DG_FP));
  cubVDs_ptr = (DG_FP *)calloc(N_max * cNp_max * Np_max, sizeof(DG_FP));
  gInterp_ptr = (DG_FP *)calloc(N_max * gNp_max * Np_max, sizeof(DG_FP));
  gFInterp0_ptr = (DG_FP *)calloc(N_max * gNfp_max * Np_max, sizeof(DG_FP));
  gFInterp1_ptr = (DG_FP *)calloc(N_max * gNfp_max * Np_max, sizeof(DG_FP));
  gFInterp2_ptr = (DG_FP *)calloc(N_max * gNfp_max * Np_max, sizeof(DG_FP));
  gF0Dr_ptr = (DG_FP *)calloc(N_max * gNfp_max * Np_max, sizeof(DG_FP));
  gF0Ds_ptr = (DG_FP *)calloc(N_max * gNfp_max * Np_max, sizeof(DG_FP));
  gF1Dr_ptr = (DG_FP *)calloc(N_max * gNfp_max * Np_max, sizeof(DG_FP));
  gF1Ds_ptr = (DG_FP *)calloc(N_max * gNfp_max * Np_max, sizeof(DG_FP));
  gF2Dr_ptr = (DG_FP *)calloc(N_max * gNfp_max * Np_max, sizeof(DG_FP));
  gF2Ds_ptr = (DG_FP *)calloc(N_max * gNfp_max * Np_max, sizeof(DG_FP));
  invMass_gInterpT_ptr = (DG_FP *)calloc(N_max * gNp_max * Np_max, sizeof(DG_FP));
  order_interp_ptr = (DG_FP *)calloc(N_max * N_max * Np_max * Np_max, sizeof(DG_FP));

  for(int N = 1; N <= N_max; N++) {
    int Np, Nfp;
    // Set num points and num face points
    DGUtils::numNodes2D(N, &Np, &Nfp);

    // Set the coordinates of the points on the refernece triangle
    arma::vec x_, y_, r_, s_;
    DGUtils::setRefXY(N, x_, y_);
    DGUtils::xy2rs(x_, y_, r_, s_);

    // Reference element matrices
    arma::mat V_ = DGUtils::vandermonde2D(r_, s_, N);
    arma::mat invV_ = arma::inv(V_);
    arma::mat MassMatrix_ = invV_.t() * invV_;
    arma::mat Dr_, Ds_;
    DGUtils::dMatrices2D(r_, s_, V_, N, Dr_, Ds_);

    // FMask
    arma::uvec fmask1_ = arma::find(arma::abs(s_ + 1)  < 1e-12);
    arma::uvec fmask2_ = arma::find(arma::abs(r_ + s_) < 1e-12);
    arma::uvec fmask3_ = arma::find(arma::abs(r_ + 1)  < 1e-12);
    fmask3_ = arma::reverse(fmask3_);
    arma::uvec fmask_  = arma::join_cols(fmask1_, fmask2_, fmask3_);

    // LIFT matrix
    arma::mat lift_ = DGUtils::lift2D(r_, s_, fmask_, V_, N);

    // Weak operators
    arma::mat Vr, Vs;
    DGUtils::gradVandermonde2D(r_, s_, N, Vr, Vs);
    arma::mat Drw_ = (V_ * Vr.t()) * arma::inv(V_ * V_.t());
    arma::mat Dsw_ = (V_ * Vs.t()) * arma::inv(V_ * V_.t());

    int intN = ceil(3.0 * N / 2.0);
    cubature(2 * (intN + 1), N, V_, invV_, Dr_, Ds_);
    // gauss(intN + 1);
    // Use same Gauss points for every order (makes p-adaptivity much more simple)
    intN = ceil(3.0 * DG_ORDER / 2.0);
    gauss(intN + 1, N, V_, invV_, Dr_, Ds_);

    arma::mat invMass = arma::inv(MassMatrix_);

    // Copy armadillo vecs and mats to global memory
    save_vec(r_ptr, r_, N, Np_max);
    save_vec(s_ptr, s_, N, Np_max);
    save_mat(v_ptr, V_, N, Np_max * Np_max);
    save_mat(invV_ptr, invV_, N, Np_max * Np_max);
    save_mat(mass_ptr, MassMatrix_, N, Np_max * Np_max);
    save_mat(invMass_ptr, invMass, N, Np_max * Np_max);
    save_mat(Dr_ptr, Dr_, N, Np_max * Np_max);
    save_mat(Ds_ptr, Ds_, N, Np_max * Np_max);
    save_mat(Drw_ptr, Drw_, N, Np_max * Np_max);
    save_mat(Dsw_ptr, Dsw_, N, Np_max * Np_max);
    save_mat(lift_ptr, lift_, N, Np_max * DG_NUM_FACES * Nfp_max);
    // memcpy(&r_ptr[(N - 1) * Np_max], r_.memptr(), r_.n_elem * sizeof(DG_FP));
    // memcpy(&s_ptr[(N - 1) * Np_max], s_.memptr(), s_.n_elem * sizeof(DG_FP));
    // memcpy(&v_ptr[(N - 1) * Np_max * Np_max], V_.memptr(), V_.n_elem * sizeof(DG_FP));
    // memcpy(&invV_ptr[(N - 1) * Np_max * Np_max], invV_.memptr(), invV_.n_elem * sizeof(DG_FP));
    // memcpy(&mass_ptr[(N - 1) * Np_max * Np_max], MassMatrix_.memptr(), MassMatrix_.n_elem * sizeof(DG_FP));
    // memcpy(&invMass_ptr[(N - 1) * Np_max * Np_max], invMass.memptr(), invMass.n_elem * sizeof(DG_FP));
    // memcpy(&Dr_ptr[(N - 1) * Np_max * Np_max], Dr_.memptr(), Dr_.n_elem * sizeof(DG_FP));
    // memcpy(&Ds_ptr[(N - 1) * Np_max * Np_max], Ds_.memptr(), Ds_.n_elem * sizeof(DG_FP));
    // memcpy(&Drw_ptr[(N - 1) * Np_max * Np_max], Drw_.memptr(), Drw_.n_elem * sizeof(DG_FP));
    // memcpy(&Dsw_ptr[(N - 1) * Np_max * Np_max], Dsw_.memptr(), Dsw_.n_elem * sizeof(DG_FP));
    // memcpy(&lift_ptr[(N - 1) * Np_max * DG_NUM_FACES * Nfp_max], lift_.memptr(), lift_.n_elem * sizeof(DG_FP));
    std::vector<int> fmask_int = arma::conv_to<std::vector<int>>::from(fmask_);
    memcpy(&FMASK[(N - 1) * DG_NUM_FACES * Nfp_max], fmask_int.data(), fmask_int.size() * sizeof(int));
    // Number of points
    DG_CONSTANTS[(N - 1) * 5 + 0] = Np;
    // Number of face points
    DG_CONSTANTS[(N - 1) * 5 + 1] = Nfp;
  }
}

void DGConstants2D::cubature(const int nCub, const int N, arma::mat &V_, arma::mat &invV_, arma::mat &Dr_, arma::mat &Ds_) {
  arma::vec c_r, c_s, cub_w_;
  DGUtils::cubature2D(nCub, c_r, c_s, cub_w_);

  arma::mat cub_V_ = DGUtils::interpMatrix2D(c_r, c_s, invV_, N);

  arma::mat cub_Dr_, cub_Ds_;
  DGUtils::dMatrices2D(c_r, c_s, V_, N, cub_Dr_, cub_Ds_);

  arma::mat cub_V_Dr = cub_V_ * Dr_;
  arma::mat cub_V_Ds = cub_V_ * Ds_;

  save_vec(cubW_g, cub_w_, N, cNp_max);
  save_mat(cubV_ptr, cub_V_, N, cNp_max * Np_max);
  save_mat(cubDr_ptr, cub_Dr_, N, cNp_max * Np_max);
  save_mat(cubDs_ptr, cub_Ds_, N, cNp_max * Np_max);
  save_mat(cubVDr_ptr, cub_V_Dr, N, cNp_max * Np_max);
  save_mat(cubVDs_ptr, cub_V_Ds, N, cNp_max * Np_max);
  // memcpy(&cubW_g[(N - 1) * cNp_max], cub_w_.memptr(), cub_w_.n_elem * sizeof(DG_FP));
  // memcpy(&cubV_ptr[(N - 1) * cNp_max * Np_max], cub_V_.memptr(), cub_V_.n_elem * sizeof(DG_FP));
  // memcpy(&cubDr_ptr[(N - 1) * cNp_max * Np_max], cub_Dr_.memptr(), cub_Dr_.n_elem * sizeof(DG_FP));
  // memcpy(&cubDs_ptr[(N - 1) * cNp_max * Np_max], cub_Ds_.memptr(), cub_Ds_.n_elem * sizeof(DG_FP));
  // memcpy(&cubVDr_ptr[(N - 1) * cNp_max * Np_max], cub_V_Dr.memptr(), cub_V_Dr.n_elem * sizeof(DG_FP));
  // memcpy(&cubVDs_ptr[(N - 1) * cNp_max * Np_max], cub_V_Ds.memptr(), cub_V_Ds.n_elem * sizeof(DG_FP));
  // Number of cubature points
  int cNp = cub_w_.n_elem;
  DG_CONSTANTS[(N - 1) * 5 + 2] = cNp;
}

void DGConstants2D::gauss(const int nGauss, const int N, arma::mat &V_, arma::mat &invV_, arma::mat &Dr_, arma::mat &Ds_) {
  arma::vec g_x, gauss_w_;
  DGUtils::jacobiGQ(0.0, 0.0, nGauss - 1, g_x, gauss_w_);

  arma::vec face1r = g_x;
  arma::vec face2r = -g_x;
  arma::vec face3r = -arma::ones<arma::vec>(nGauss);
  arma::vec face1s = -arma::ones<arma::vec>(nGauss);
  arma::vec face2s = g_x;
  arma::vec face3s = -g_x;

  arma::mat gauss_interp1_ = DGUtils::vandermonde2D(face1r, face1s, N) * invV_;
  arma::mat gauss_interp2_ = DGUtils::vandermonde2D(face2r, face2s, N) * invV_;
  arma::mat gauss_interp3_ = DGUtils::vandermonde2D(face3r, face3s, N) * invV_;

  arma::mat gauss_interp_ = arma::join_vert(gauss_interp1_, gauss_interp2_, gauss_interp3_);

  arma::mat gauss_i1_Dr = gauss_interp1_ * Dr_;
  arma::mat gauss_i1_Ds = gauss_interp1_ * Ds_;
  arma::mat gauss_i2_Dr = gauss_interp2_ * Dr_;
  arma::mat gauss_i2_Ds = gauss_interp2_ * Ds_;
  arma::mat gauss_i3_Dr = gauss_interp3_ * Dr_;
  arma::mat gauss_i3_Ds = gauss_interp3_ * Ds_;
  // arma::mat invMass_gauss_interpT = invMass * gauss_interp_.t();
  arma::mat invMass_gauss_interpT = V_ * V_.t() * gauss_interp_.t();

  save_vec(gaussW_g, gauss_w_, N, gNfp_max);
  save_mat(gInterp_ptr, gauss_interp_, N, gNp_max * Np_max);
  save_mat(gFInterp0_ptr, gauss_interp1_, N, gNfp_max * Np_max);
  save_mat(gFInterp1_ptr, gauss_interp2_, N, gNfp_max * Np_max);
  save_mat(gFInterp2_ptr, gauss_interp3_, N, gNfp_max * Np_max);
  save_mat(gF0Dr_ptr, gauss_i1_Dr, N, gNfp_max * Np_max);
  save_mat(gF0Ds_ptr, gauss_i1_Ds, N, gNfp_max * Np_max);
  save_mat(gF1Dr_ptr, gauss_i2_Dr, N, gNfp_max * Np_max);
  save_mat(gF1Ds_ptr, gauss_i2_Ds, N, gNfp_max * Np_max);
  save_mat(gF2Dr_ptr, gauss_i3_Dr, N, gNfp_max * Np_max);
  save_mat(gF2Ds_ptr, gauss_i3_Ds, N, gNfp_max * Np_max);
  save_mat(invMass_gInterpT_ptr, invMass_gauss_interpT, N, gNp_max * Np_max);
  // memcpy(&gaussW_g[(N - 1) * gNfp_max], gauss_w_.memptr(), gauss_w_.n_elem * sizeof(DG_FP));
  // memcpy(&gInterp_ptr[(N - 1) * gNp_max * Np_max], gauss_interp_.memptr(), gauss_interp_.n_elem * sizeof(DG_FP));
  // memcpy(&gFInterp0_ptr[(N - 1) * gNfp_max * Np_max], gauss_interp1_.memptr(), gauss_interp1_.n_elem * sizeof(DG_FP));
  // memcpy(&gFInterp1_ptr[(N - 1) * gNfp_max * Np_max], gauss_interp2_.memptr(), gauss_interp2_.n_elem * sizeof(DG_FP));
  // memcpy(&gFInterp2_ptr[(N - 1) * gNfp_max * Np_max], gauss_interp3_.memptr(), gauss_interp3_.n_elem * sizeof(DG_FP));
  // memcpy(&gF0Dr_ptr[(N - 1) * gNfp_max * Np_max], gauss_i1_Dr.memptr(), gauss_i1_Dr.n_elem * sizeof(DG_FP));
  // memcpy(&gF0Ds_ptr[(N - 1) * gNfp_max * Np_max], gauss_i1_Ds.memptr(), gauss_i1_Ds.n_elem * sizeof(DG_FP));
  // memcpy(&gF1Dr_ptr[(N - 1) * gNfp_max * Np_max], gauss_i2_Dr.memptr(), gauss_i2_Dr.n_elem * sizeof(DG_FP));
  // memcpy(&gF1Ds_ptr[(N - 1) * gNfp_max * Np_max], gauss_i2_Ds.memptr(), gauss_i2_Ds.n_elem * sizeof(DG_FP));
  // memcpy(&gF2Dr_ptr[(N - 1) * gNfp_max * Np_max], gauss_i3_Dr.memptr(), gauss_i3_Dr.n_elem * sizeof(DG_FP));
  // memcpy(&gF2Ds_ptr[(N - 1) * gNfp_max * Np_max], gauss_i3_Ds.memptr(), gauss_i3_Ds.n_elem * sizeof(DG_FP));
  // memcpy(&invMass_gInterpT_ptr[(N - 1) * gNp_max * Np_max], invMass_gauss_interpT.memptr(), invMass_gauss_interpT.n_elem * sizeof(DG_FP));
  // Number of gauss points
  int gNp = gauss_w_.n_elem * 3;
  DG_CONSTANTS[(N - 1) * 5 + 3] = gNp;
  // Number of gauss points per edge
  int gNfp = gauss_w_.n_elem;
  DG_CONSTANTS[(N - 1) * 5 + 4] = gNfp;
}

void DGConstants2D::calc_interp_mats() {
  for(int n0 = 1; n0 <= N_max; n0++) {
    arma::vec x_n0, y_n0, r_n0, s_n0;
    DGUtils::setRefXY(n0, x_n0, y_n0);
    DGUtils::xy2rs(x_n0, y_n0, r_n0, s_n0);
    arma::mat V_n0 = DGUtils::vandermonde2D(r_n0, s_n0, n0);
    arma::mat invV_n0 = arma::inv(V_n0);
    for(int n1 = 1; n1 <= N_max; n1++) {
      if(n0 != n1) {
        arma::vec x_n1, y_n1, r_n1, s_n1;
        DGUtils::setRefXY(n1, x_n1, y_n1);
        DGUtils::xy2rs(x_n1, y_n1, r_n1, s_n1);
        arma::mat interp_ = DGUtils::interpMatrix2D(r_n1, s_n1, invV_n0, n0);
        arma::Mat<DG_FP> interp_2 = arma::conv_to<arma::Mat<DG_FP>>::from(interp_);
        memcpy(&order_interp_ptr[((n0 - 1) * N_max + (n1 - 1)) * Np_max * Np_max], interp_2.memptr(), interp_2.n_elem * sizeof(DG_FP));
      }
    }
  }
}

DG_FP* DGConstants2D::get_mat_ptr(Constant_Matrix matrix) {
  switch(matrix) {
    case R:
      return r_ptr;
    case S:
      return s_ptr;
    case V:
      return v_ptr;
    case INV_V:
      return invV_ptr;
    case MASS:
      return mass_ptr;
    case INV_MASS:
      return invMass_ptr;
    case DR:
      return Dr_ptr;
    case DS:
      return Ds_ptr;
    case DRW:
      return Drw_ptr;
    case DSW:
      return Dsw_ptr;
    case LIFT:
      return lift_ptr;
    case CUB_V:
      return cubV_ptr;
    case CUB_DR:
      return cubDr_ptr;
    case CUB_DS:
      return cubDs_ptr;
    case CUB_VDR:
      return cubVDr_ptr;
    case CUB_VDS:
      return cubVDs_ptr;
    case GAUSS_INTERP:
      return gInterp_ptr;
    case GAUSS_FINTERP0:
      return gFInterp0_ptr;
    case GAUSS_FINTERP1:
      return gFInterp1_ptr;
    case GAUSS_FINTERP2:
      return gFInterp2_ptr;
    case GAUSS_F0DR:
      return gF0Dr_ptr;
    case GAUSS_F0DS:
      return gF0Ds_ptr;
    case GAUSS_F1DR:
      return gF1Dr_ptr;
    case GAUSS_F1DS:
      return gF1Ds_ptr;
    case GAUSS_F2DR:
      return gF2Dr_ptr;
    case GAUSS_F2DS:
      return gF2Ds_ptr;
    case INV_MASS_GAUSS_INTERP_T:
      return invMass_gInterpT_ptr;
    case INTERP_MATRIX_ARRAY:
      return order_interp_ptr;
    default:
      throw std::runtime_error("This constant matrix is not supported by DGConstants2D\n");
      return nullptr;
  }
}

DGConstants2D::~DGConstants2D() {
  delete r_ptr;
  delete s_ptr;
  delete v_ptr;
  delete invV_ptr;
  delete mass_ptr;
  delete invMass_ptr;
  delete Dr_ptr;
  delete Ds_ptr;
  delete Drw_ptr;
  delete Dsw_ptr;
  delete lift_ptr;
  delete cubV_ptr;
  delete cubDr_ptr;
  delete cubDs_ptr;
  delete cubVDr_ptr;
  delete cubVDs_ptr;
  delete gInterp_ptr;
  delete gFInterp0_ptr;
  delete gFInterp1_ptr;
  delete gFInterp2_ptr;
  delete gF0Dr_ptr;
  delete gF0Ds_ptr;
  delete gF1Dr_ptr;
  delete gF1Ds_ptr;
  delete gF2Dr_ptr;
  delete gF2Ds_ptr;
  delete invMass_gInterpT_ptr;
  delete order_interp_ptr;
}
