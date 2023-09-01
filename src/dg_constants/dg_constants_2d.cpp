#include "dg_constants/dg_constants_2d.h"

#include <stdexcept>

#include "dg_compiler_defs.h"
#include "dg_utils.h"

int FMASK[DG_ORDER * DG_NUM_FACES * DG_NPF];
int DG_CONSTANTS[DG_ORDER * 5];
DG_FP cubW_g[DG_ORDER * DG_CUB_NP];
DG_FP gaussW_g[DG_ORDER * DG_GF_NP];

int FMASK_TK[DG_ORDER * DG_NUM_FACES * DG_NPF];
int DG_CONSTANTS_TK[DG_ORDER * 5];
DG_FP cubW_g_TK[DG_ORDER * DG_CUB_NP];
DG_FP gaussW_g_TK[DG_ORDER * DG_GF_NP];

void save_mat(DG_FP *mem_ptr, arma::mat &mat, const int N, const int max_size) {
  #ifdef DG_COL_MAJ
  arma::Mat<DG_FP> mat_2 = arma::conv_to<arma::Mat<DG_FP>>::from(mat);
  #else
  arma::Mat<DG_FP> mat_2 = arma::conv_to<arma::Mat<DG_FP>>::from(mat.t());
  #endif
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
  // cNp_max = DG_CUB_NP;
  // Set max num of Gauss points
  // gNfp_max = ceil(3.0 * DG_ORDER / 2.0) + 1;
  // gNp_max = 3 * gNfp_max;

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
  mmF0_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  mmF1_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  mmF2_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  eMat_ptr = (DG_FP *)calloc(N_max * DG_NUM_FACES * Nfp_max * Np_max, sizeof(DG_FP));
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
    // std::cout << "Dr_" << std::endl;
    // std::cout << Dr_ << std::endl;
    // std::cout << "Ds_" << std::endl;
    // std::cout << Ds_ << std::endl;

    // FMask
    arma::uvec fmask1_ = arma::find(arma::abs(s_ + 1)  < 1e-12);
    arma::uvec fmask2_ = arma::find(arma::abs(r_ + s_) < 1e-12);
    arma::uvec fmask3_ = arma::find(arma::abs(r_ + 1)  < 1e-12);
    fmask3_ = arma::reverse(fmask3_);
    arma::uvec fmask_  = arma::join_cols(fmask1_, fmask2_, fmask3_);

    // LIFT matrix
    arma::mat eMat_ = DGUtils::eMat2D(r_, s_, fmask_, N);
    arma::mat lift_ = DGUtils::lift2D(r_, s_, fmask_, V_, N);
    arma::mat mmF0_, mmF1_, mmF2_;
    DGUtils::faceMassMatrix2D(r_, s_, fmask_, V_, N, mmF0_, mmF1_, mmF2_);

    // Weak operators
    arma::mat Vr, Vs;
    DGUtils::gradVandermonde2D(r_, s_, N, Vr, Vs);
    arma::mat Drw_ = (V_ * Vr.t()) * arma::inv(V_ * V_.t());
    arma::mat Dsw_ = (V_ * Vs.t()) * arma::inv(V_ * V_.t());
    arma::mat invMass = arma::inv(MassMatrix_);
    // std::cout << "Drw_" << std::endl;
    // std::cout << Drw_ << std::endl;
    // std::cout << "Dsw_" << std::endl;
    // std::cout << Dsw_ << std::endl;

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
    save_mat(eMat_ptr, eMat_, N, Np_max * DG_NUM_FACES * Nfp_max);
    save_mat(lift_ptr, lift_, N, Np_max * DG_NUM_FACES * Nfp_max);
    save_mat(mmF0_ptr, mmF0_, N, Np_max * Np_max);
    save_mat(mmF1_ptr, mmF1_, N, Np_max * Np_max);
    save_mat(mmF2_ptr, mmF2_, N, Np_max * Np_max);
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
    memcpy(&FMASK_TK[(N - 1) * DG_NUM_FACES * Nfp_max], fmask_int.data(), fmask_int.size() * sizeof(int));
    // Number of points
    DG_CONSTANTS[(N - 1) * 5 + 0] = Np;
    // Number of face points
    DG_CONSTANTS[(N - 1) * 5 + 1] = Nfp;
    // Number of points
    DG_CONSTANTS_TK[(N - 1) * 5 + 0] = Np;
    // Number of face points
    DG_CONSTANTS_TK[(N - 1) * 5 + 1] = Nfp;
  }

  cubature(2 * DG_ORDER);
  gauss(2 * DG_ORDER);
}

void DGConstants2D::cubature(const int nCub) {
  arma::vec x_, y_, r_, s_;
  DGUtils::setRefXY(DG_ORDER, x_, y_);
  DGUtils::xy2rs(x_, y_, r_, s_);
  arma::mat V_ = DGUtils::vandermonde2D(r_, s_, DG_ORDER);
  arma::mat invV_ = arma::inv(V_);

  // arma::vec cub_r, cub_s, cub_w;
  // DGUtils::cubature2D(2 * DG_ORDER, cub_r, cub_s, cub_w);
  const double cubTriR6[12] = {-5.611400349004336e-01, 1.222800698008678e-01,-5.611400349004341e-01,-3.972407177556988e-02,-3.972407177557002e-02,-9.205518564488601e-01,-7.167619681520639e-01, 6.780185194295818e-01,-9.612565512775184e-01, 6.780185194295824e-01,-7.167619681520634e-01,-9.612565512775182e-01};
  const double cubTriS6[12] = {-5.611400349004347e-01,-5.611400349004335e-01, 1.222800698008681e-01,-9.205518564488602e-01,-3.972407177556986e-02,-3.972407177556986e-02,-9.612565512775185e-01,-7.167619681520636e-01, 6.780185194295830e-01,-9.612565512775185e-01, 6.780185194295830e-01,-7.167619681520636e-01};
  const double cubTriW6[12] = { 3.426662483059627e-01, 3.426662483059627e-01, 3.426662483059627e-01, 1.614621791860623e-01, 1.614621791860623e-01, 1.614621791860623e-01, 8.126911958732136e-02, 8.126911958732136e-02, 8.126911958732136e-02, 8.126911958732136e-02, 8.126911958732136e-02, 8.126911958732136e-02};

  arma::vec cub_r(cubTriR6, 12);
  arma::vec cub_s(cubTriS6, 12);
  arma::vec cub_w(cubTriW6, 12);

  arma::mat cubInterp = DGUtils::interpMatrix2D(cub_r, cub_s, invV_, DG_ORDER);
  arma::mat cubProj   = DGUtils::cubaturePMat2D(r_, s_, cub_r, cub_s, DG_ORDER);
  arma::mat cubPDrT, cubPDsT;
  DGUtils::cubaturePDwMat2D(r_, s_, cub_r, cub_s, DG_ORDER, cubPDrT, cubPDsT);
  const int cubNp = cub_r.n_elem;
  arma::mat diag_w(cubNp, cubNp);
  diag_w.zeros();
  for(int i = 0; i < cubNp; i++) {
    diag_w(i,i) = cub_w(i);
  }

  printf("DG_CUB_2D_NP vs %d\n", cubNp);

  // std::cout << "Test" << std::endl;
  // std::cout << cubProj * diag_w * cubInterp << std::endl;

  // arma::mat tmp_cub_ident = cubProj * diag_w * cubInterp;
  // save_mat(Drw_ptr, tmp_cub_ident, DG_ORDER, DG_NP * DG_NP);

  // std::cout << "cubPDrT" << std::endl;
  // std::cout << cubPDrT << std::endl;

  // std::cout << "cubPDsT" << std::endl;
  // std::cout << cubPDsT << std::endl;

  // std::cout << "diag_w" << std::endl;
  // std::cout << diag_w << std::endl;

  cubProj = cubProj * diag_w;
  cubPDrT = cubPDrT * diag_w;
  cubPDsT = cubPDsT * diag_w;

  // std::cout << "cubInterp" << std::endl;
  // std::cout << cubInterp << std::endl;

  // std::cout << "cubProj" << std::endl;
  // std::cout << cubProj << std::endl;

  // std::cout << "cubPDrT" << std::endl;
  // std::cout << cubPDrT << std::endl;

  // std::cout << "cubPDsT" << std::endl;
  // std::cout << cubPDsT << std::endl;

  cub_r_ptr = (DG_FP *)calloc(DG_CUB_2D_NP, sizeof(DG_FP));
  cub_s_ptr = (DG_FP *)calloc(DG_CUB_2D_NP, sizeof(DG_FP));
  cub_w_ptr = (DG_FP *)calloc(DG_CUB_2D_NP, sizeof(DG_FP));
  cubInterp_ptr = (DG_FP *)calloc(DG_NP * DG_CUB_2D_NP, sizeof(DG_FP));
  cubProj_ptr = (DG_FP *)calloc(DG_NP * DG_CUB_2D_NP, sizeof(DG_FP));
  cubPDrT_ptr = (DG_FP *)calloc(DG_NP * DG_CUB_2D_NP, sizeof(DG_FP));
  cubPDsT_ptr = (DG_FP *)calloc(DG_NP * DG_CUB_2D_NP, sizeof(DG_FP));

  save_vec(cub_r_ptr, cub_r, 1, DG_CUB_2D_NP);
  save_vec(cub_s_ptr, cub_s, 1, DG_CUB_2D_NP);
  save_vec(cub_w_ptr, cub_w, 1, DG_CUB_2D_NP);
  save_mat(cubInterp_ptr, cubInterp, 1, DG_NP * DG_CUB_2D_NP);
  save_mat(cubProj_ptr, cubProj, 1, DG_NP * DG_CUB_2D_NP);
  save_mat(cubPDrT_ptr, cubPDrT, 1, DG_NP * DG_CUB_2D_NP);
  save_mat(cubPDsT_ptr, cubPDsT, 1, DG_NP * DG_CUB_2D_NP);
}

void DGConstants2D::gauss(const int nGauss) {
  arma::vec x_, y_, r_, s_;
  DGUtils::setRefXY(DG_ORDER, x_, y_);
  DGUtils::xy2rs(x_, y_, r_, s_);
  arma::mat V_ = DGUtils::vandermonde2D(r_, s_, DG_ORDER);
  arma::mat invV_ = arma::inv(V_);
  // FMask
  arma::uvec fmask1_ = arma::find(arma::abs(s_ + 1)  < 1e-12);
  arma::uvec fmask2_ = arma::find(arma::abs(r_ + s_) < 1e-12);
  arma::uvec fmask3_ = arma::find(arma::abs(r_ + 1)  < 1e-12);
  fmask3_ = arma::reverse(fmask3_);
  arma::uvec fmask_  = arma::join_cols(fmask1_, fmask2_, fmask3_);

  arma::vec g_x, gauss_w_;
  DGUtils::jacobiGQ(0.0, 0.0, nGauss, g_x, gauss_w_);

  const int npf_cub = gauss_w_.n_elem;

  arma::vec face1r = g_x;
  arma::vec face2r = -g_x;
  arma::vec face3r = -arma::ones<arma::vec>(npf_cub);
  arma::vec face1s = -arma::ones<arma::vec>(npf_cub);
  arma::vec face2s = g_x;
  arma::vec face3s = -g_x;

  printf("DG_CUB_SURF_2D_NP vs %d\n", npf_cub);

  arma::vec interp_r(npf_cub * 3);
  arma::vec interp_s(npf_cub * 3);
  arma::vec interp_w(npf_cub * 3);
  for(int i = 0; i < npf_cub; i++) {
    interp_r[i]               = face1r[i];
    interp_r[i + npf_cub]     = face2r[i];
    interp_r[i + 2 * npf_cub] = face3r[i];
    interp_s[i]               = face1s[i];
    interp_s[i + npf_cub]     = face2s[i];
    interp_s[i + 2 * npf_cub] = face3s[i];
    interp_w[i]               = gauss_w_[i];
    interp_w[i + npf_cub]     = gauss_w_[i];
    interp_w[i + 2 * npf_cub] = gauss_w_[i];
  }

  arma::mat tmp_interp_ = DGUtils::interpMatrix2D(interp_r, interp_s, invV_, DG_ORDER);

  arma::mat gauss_interp_;
  gauss_interp_.zeros(npf_cub * 3, DG_NPF * 3);
  for(int face = 0; face < 3; face++) {
    for(int cub_pt = 0; cub_pt < npf_cub; cub_pt++) {
      for(int nodal_pt = 0; nodal_pt < DG_NPF; nodal_pt++) {
        gauss_interp_(face * npf_cub + cub_pt, face * DG_NPF + nodal_pt) = tmp_interp_(face * npf_cub + cub_pt, fmask_(face * DG_NPF + nodal_pt));
      }
    }
  }

  // arma::mat gauss_interp1_ = DGUtils::vandermonde2D(face1r, face1s, DG_ORDER) * invV_;
  // arma::mat gauss_interp2_ = DGUtils::vandermonde2D(face2r, face2s, DG_ORDER) * invV_;
  // arma::mat gauss_interp3_ = DGUtils::vandermonde2D(face3r, face3s, DG_ORDER) * invV_;

  // arma::mat gauss_interp_ = arma::join_vert(gauss_interp1_, gauss_interp2_, gauss_interp3_);

  arma::mat diagW(npf_cub * 3, npf_cub * 3);
  diagW.zeros();
  for(int i = 0; i < npf_cub * 3; i++) {
    diagW(i,i) = interp_w[i];
  }

  arma::mat gauss_lift_ = V_ * V_.t() * tmp_interp_.t() * diagW;

  cubInterpSurf_ptr = (DG_FP *)calloc(DG_NUM_FACES * DG_NPF * DG_NUM_FACES * DG_CUB_SURF_2D_NP, sizeof(DG_FP));
  cubLiftSurf_ptr   = (DG_FP *)calloc(DG_NP * DG_NUM_FACES * DG_CUB_SURF_2D_NP, sizeof(DG_FP));

  save_mat(cubInterpSurf_ptr, gauss_interp_, 1, DG_NUM_FACES * DG_NPF * DG_NUM_FACES * DG_CUB_SURF_2D_NP);
  save_mat(cubLiftSurf_ptr, gauss_lift_, 1, DG_NP * DG_NUM_FACES * DG_CUB_SURF_2D_NP);

  int gNp = gauss_w_.n_elem * 3;
  DG_CONSTANTS[(DG_ORDER - 1) * 5 + 3] = gNp;
  DG_CONSTANTS_TK[(DG_ORDER - 1) * 5 + 3] = gNp;
  // Number of gauss points per edge
  int gNfp = gauss_w_.n_elem;
  DG_CONSTANTS[(DG_ORDER - 1) * 5 + 4] = gNfp;
  DG_CONSTANTS_TK[(DG_ORDER - 1) * 5 + 4] = gNfp;
}

/*
void DGConstants2D::cubature(const int nCub, const int N, arma::mat &V_, arma::mat &invV_, arma::mat &Dr_, arma::mat &Ds_) {
  arma::vec c_r, c_s, cub_w_;
  DGUtils::cubature2D(nCub, c_r, c_s, cub_w_);
  arma::mat cub_V_ = DGUtils::interpMatrix2D(c_r, c_s, invV_, N);

  arma::mat cub_Dr_, cub_Ds_;
  DGUtils::dMatrices2D(c_r, c_s, V_, N, cub_Dr_, cub_Ds_);

  arma::mat cub_V_Dr = cub_V_ * Dr_;
  arma::mat cub_V_Ds = cub_V_ * Ds_;

  save_vec(cubW_g, cub_w_, N, cNp_max);
  save_vec(cubW_g_TK, cub_w_, N, cNp_max);
  save_mat(cubV_ptr, cub_V_, N, cNp_max * Np_max);
  save_mat(cubDr_ptr, cub_Dr_, N, cNp_max * Np_max);
  save_mat(cubDs_ptr, cub_Ds_, N, cNp_max * Np_max);
  save_mat(cubVDr_ptr, cub_V_Dr, N, cNp_max * Np_max);
  save_mat(cubVDs_ptr, cub_V_Ds, N, cNp_max * Np_max);

  // Number of cubature points
  int cNp = cub_w_.n_elem;
  DG_CONSTANTS[(N - 1) * 5 + 2] = cNp;
  DG_CONSTANTS_TK[(N - 1) * 5 + 2] = cNp;
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
  save_vec(gaussW_g_TK, gauss_w_, N, gNfp_max);
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

  // Number of gauss points
  int gNp = gauss_w_.n_elem * 3;
  DG_CONSTANTS[(N - 1) * 5 + 3] = gNp;
  DG_CONSTANTS_TK[(N - 1) * 5 + 3] = gNp;
  // Number of gauss points per edge
  int gNfp = gauss_w_.n_elem;
  DG_CONSTANTS[(N - 1) * 5 + 4] = gNfp;
  DG_CONSTANTS_TK[(N - 1) * 5 + 4] = gNfp;
}
*/

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
        #ifdef DG_COL_MAJ
        arma::Mat<DG_FP> interp_2 = arma::conv_to<arma::Mat<DG_FP>>::from(interp_);
        #else
        arma::Mat<DG_FP> interp_2 = arma::conv_to<arma::Mat<DG_FP>>::from(interp_.t());
        #endif
        memcpy(&order_interp_ptr[((n0 - 1) * N_max + (n1 - 1)) * Np_max * Np_max], interp_2.memptr(), interp_2.n_elem * sizeof(DG_FP));
      }
    }
  }

  Dr_ptr_sp = (float *)calloc(N_max * Np_max * Np_max, sizeof(float));
  Ds_ptr_sp = (float *)calloc(N_max * Np_max * Np_max, sizeof(float));
  Drw_ptr_sp = (float *)calloc(N_max * Np_max * Np_max, sizeof(float));
  Dsw_ptr_sp = (float *)calloc(N_max * Np_max * Np_max, sizeof(float));
  mass_ptr_sp = (float *)calloc(N_max * Np_max * Np_max, sizeof(float));
  invMass_ptr_sp = (float *)calloc(N_max * Np_max * Np_max, sizeof(float));
  invV_ptr_sp = (float *)calloc(N_max * Np_max * Np_max, sizeof(float));
  v_ptr_sp = (float *)calloc(N_max * Np_max * Np_max, sizeof(float));
  lift_ptr_sp = (float *)calloc(N_max * DG_NUM_FACES * Nfp_max * Np_max, sizeof(float));
  eMat_ptr_sp = (float *)calloc(N_max * DG_NUM_FACES * Nfp_max * Np_max, sizeof(float));
  order_interp_ptr_sp = (float *)calloc(N_max * N_max * Np_max * Np_max, sizeof(float));

  for(int i = 0; i < N_max * Np_max * Np_max; i++) {
    Dr_ptr_sp[i] = (float)Dr_ptr[i];
    Ds_ptr_sp[i] = (float)Ds_ptr[i];
    Drw_ptr_sp[i] = (float)Drw_ptr[i];
    Dsw_ptr_sp[i] = (float)Dsw_ptr[i];
    mass_ptr_sp[i] = (float)mass_ptr[i];
    invMass_ptr_sp[i] = (float)invMass_ptr[i];
    invV_ptr_sp[i] = (float)invV_ptr[i];
    v_ptr_sp[i] = (float)v_ptr[i];
  }

  for(int i = 0; i < N_max * DG_NUM_FACES * Nfp_max * Np_max; i++) {
    lift_ptr_sp[i] = (float)lift_ptr[i];
    eMat_ptr_sp[i] = (float)eMat_ptr[i];
  }

  for(int i = 0; i < N_max * N_max * Np_max * Np_max; i++) {
    order_interp_ptr_sp[i] = (float)order_interp_ptr[i];
  }

  transfer_kernel_ptrs();
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
    case MM_F0:
      return mmF0_ptr;
    case MM_F1:
      return mmF1_ptr;
    case MM_F2:
      return mmF2_ptr;
    case EMAT:
      return eMat_ptr;
    case INTERP_MATRIX_ARRAY:
      return order_interp_ptr;
    case CUB2D_R:
      return cub_r_ptr;
    case CUB2D_S:
      return cub_s_ptr;
    case CUB2D_W:
      return cub_w_ptr;
    case CUB2D_INTERP:
      return cubInterp_ptr;
    case CUB2D_PROJ:
      return cubProj_ptr;
    case CUB2D_PDR:
      return cubPDrT_ptr;
    case CUB2D_PDS:
      return cubPDsT_ptr;
    case CUBSURF2D_INTERP:
      return cubInterpSurf_ptr;
    case CUBSURF2D_LIFT:
      return cubLiftSurf_ptr;
    default:
      throw std::runtime_error("This constant matrix is not supported by DGConstants2D\n");
      return nullptr;
  }
}

DGConstants2D::~DGConstants2D() {
  clean_up_kernel_ptrs();

  free(r_ptr);
  free(s_ptr);
  free(v_ptr);
  free(invV_ptr);
  free(mass_ptr);
  free(invMass_ptr);
  free(Dr_ptr);
  free(Ds_ptr);
  free(Drw_ptr);
  free(Dsw_ptr);
  free(lift_ptr);
  free(cub_r_ptr);
  free(cub_s_ptr);
  free(cub_w_ptr);
  free(cubInterp_ptr);
  free(cubProj_ptr);
  free(cubPDrT_ptr);
  free(cubPDsT_ptr);
  free(cubInterpSurf_ptr);
  free(cubLiftSurf_ptr);
  free(mmF0_ptr);
  free(mmF1_ptr);
  free(mmF2_ptr);
  free(eMat_ptr);
  free(order_interp_ptr);

  free(Dr_ptr_sp);
  free(Ds_ptr_sp);
  free(Drw_ptr_sp);
  free(Dsw_ptr_sp);
  free(mass_ptr_sp);
  free(invMass_ptr_sp);
  free(invV_ptr_sp);
  free(v_ptr_sp);
  free(lift_ptr_sp);
  free(eMat_ptr_sp);
  free(order_interp_ptr_sp);
}
