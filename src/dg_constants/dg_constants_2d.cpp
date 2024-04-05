#include "dg_constants/dg_constants_2d.h"

#include "dg_abort.h"
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

  // Create matrices of all orders
  r_ptr = (DG_FP *)calloc(N_max * Np_max, sizeof(DG_FP));
  s_ptr = (DG_FP *)calloc(N_max * Np_max, sizeof(DG_FP));
  order_interp_ptr = (DG_FP *)calloc(N_max * N_max * Np_max * Np_max, sizeof(DG_FP));
  
  dg_mats.insert({V, new DGConstantMatrix(Np_max, Np_max, true)});
  dg_mats.insert({INV_V, new DGConstantMatrix(Np_max, Np_max, true)});
  dg_mats.insert({MASS, new DGConstantMatrix(Np_max, Np_max, true)});
  dg_mats.insert({INV_MASS, new DGConstantMatrix(Np_max, Np_max, true)});
  dg_mats.insert({DR, new DGConstantMatrix(Np_max, Np_max, true)});
  dg_mats.insert({DS, new DGConstantMatrix(Np_max, Np_max, true)});
  dg_mats.insert({DRW, new DGConstantMatrix(Np_max, Np_max, true)});
  dg_mats.insert({DSW, new DGConstantMatrix(Np_max, Np_max, true)});
  dg_mats.insert({EMAT, new DGConstantMatrix(Np_max, DG_NUM_FACES * Nfp_max, true)});
  dg_mats.insert({LIFT, new DGConstantMatrix(Np_max, DG_NUM_FACES * Nfp_max, true)});
  dg_mats.insert({MM_F0, new DGConstantMatrix(Np_max, Np_max, true)});
  dg_mats.insert({MM_F1, new DGConstantMatrix(Np_max, Np_max, true)});
  dg_mats.insert({MM_F2, new DGConstantMatrix(Np_max, Np_max, true)});

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

    // Copy armadillo vecs and mats to DGConstantMatrix objects
    save_vec(r_ptr, r_, N, Np_max);
    save_vec(s_ptr, s_, N, Np_max);
    dg_mats.at(V)->set_mat(V_, N);
    dg_mats.at(INV_V)->set_mat(invV_, N);
    dg_mats.at(MASS)->set_mat(MassMatrix_, N);
    dg_mats.at(INV_MASS)->set_mat(invMass, N);
    dg_mats.at(DR)->set_mat(Dr_, N);
    dg_mats.at(DS)->set_mat(Ds_, N);
    dg_mats.at(DRW)->set_mat(Drw_, N);
    dg_mats.at(DSW)->set_mat(Dsw_, N);
    dg_mats.at(EMAT)->set_mat(eMat_, N);
    dg_mats.at(LIFT)->set_mat(lift_, N);
    dg_mats.at(MM_F0)->set_mat(mmF0_, N);
    dg_mats.at(MM_F1)->set_mat(mmF1_, N);
    dg_mats.at(MM_F2)->set_mat(mmF2_, N);
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

  // Get cubature points in R-S coordinates and cubature weights
  arma::vec cub_r, cub_s, cub_w;
  DGUtils::cubature2D(nCub, cub_r, cub_s, cub_w);

  // Matrix to interpolate to the cubature points
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

  // printf("DG_CUB_2D_NP vs %d\n", cubNp);

  // Matrix to project back to DG points
  cubProj = cubProj * diag_w;
  // Matrices to project back to DG points while taking the gradient in R-S coordinates
  cubPDrT = cubPDrT * diag_w;
  cubPDsT = cubPDsT * diag_w;

  cub_r_ptr = (DG_FP *)calloc(DG_CUB_2D_NP, sizeof(DG_FP));
  cub_s_ptr = (DG_FP *)calloc(DG_CUB_2D_NP, sizeof(DG_FP));
  cub_w_ptr = (DG_FP *)calloc(DG_CUB_2D_NP, sizeof(DG_FP));
  dg_mats.insert({CUB2D_INTERP, new DGConstantMatrix(DG_CUB_2D_NP, DG_NP, false)});
  dg_mats.insert({CUB2D_PROJ, new DGConstantMatrix(DG_NP, DG_CUB_2D_NP, false)});
  dg_mats.insert({CUB2D_PDR, new DGConstantMatrix(DG_NP, DG_CUB_2D_NP, false)});
  dg_mats.insert({CUB2D_PDS, new DGConstantMatrix(DG_NP, DG_CUB_2D_NP, false)});

  save_vec(cub_r_ptr, cub_r, 1, DG_CUB_2D_NP);
  save_vec(cub_s_ptr, cub_s, 1, DG_CUB_2D_NP);
  save_vec(cub_w_ptr, cub_w, 1, DG_CUB_2D_NP);
  dg_mats.at(CUB2D_INTERP)->set_mat(cubInterp);
  dg_mats.at(CUB2D_PROJ)->set_mat(cubProj);
  dg_mats.at(CUB2D_PDR)->set_mat(cubPDrT);
  dg_mats.at(CUB2D_PDS)->set_mat(cubPDsT);
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

  // printf("DG_CUB_SURF_2D_NP vs %d\n", npf_cub);

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

  // Matrix to interpolate from the DG surface points to the cubature surface points
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

  // Lift matrix while going from surface cubature points to DG points
  arma::mat gauss_lift_ = V_ * V_.t() * tmp_interp_.t() * diagW;

  dg_mats.insert({CUBSURF2D_INTERP, new DGConstantMatrix(DG_CUB_SURF_2D_NP * DG_NUM_FACES, DG_NUM_FACES * DG_NPF, false)});
  dg_mats.insert({CUBSURF2D_LIFT, new DGConstantMatrix(DG_NP, DG_CUB_SURF_2D_NP * DG_NUM_FACES, false)});

  dg_mats.at(CUBSURF2D_INTERP)->set_mat(gauss_interp_);
  dg_mats.at(CUBSURF2D_LIFT)->set_mat(gauss_lift_);

  int gNp = gauss_w_.n_elem * 3;
  DG_CONSTANTS[(DG_ORDER - 1) * 5 + 3] = gNp;
  DG_CONSTANTS_TK[(DG_ORDER - 1) * 5 + 3] = gNp;
  // Number of gauss points per edge
  int gNfp = gauss_w_.n_elem;
  DG_CONSTANTS[(DG_ORDER - 1) * 5 + 4] = gNfp;
  DG_CONSTANTS_TK[(DG_ORDER - 1) * 5 + 4] = gNfp;
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
        #ifdef DG_COL_MAJ
        arma::Mat<DG_FP> interp_2 = arma::conv_to<arma::Mat<DG_FP>>::from(interp_);
        #else
        arma::Mat<DG_FP> interp_2 = arma::conv_to<arma::Mat<DG_FP>>::from(interp_.t());
        #endif
        memcpy(&order_interp_ptr[((n0 - 1) * N_max + (n1 - 1)) * Np_max * Np_max], interp_2.memptr(), interp_2.n_elem * sizeof(DG_FP));
      }
    }
  }

  // Convert interpolation matrices to single-precision
  order_interp_ptr_sp = (float *)calloc(N_max * N_max * Np_max * Np_max, sizeof(float));
  for(int i = 0; i < N_max * N_max * Np_max * Np_max; i++) {
    order_interp_ptr_sp[i] = (float)order_interp_ptr[i];
  }

  // Copy all DGConstantMatrix matrices to device memory.
  for(const auto &dg_mat : dg_mats) {
    dg_mat.second->transfer_to_device();
  }

  transfer_kernel_ptrs();
}

// Get a pointer to the constant matrix (host memory).
// For matrices with multiple DG orders, the pointer will point to the 1st order matrix.
DG_FP* DGConstants2D::get_mat_ptr(Constant_Matrix matrix) {
  switch(matrix) {
    case R:
      return r_ptr;
    case S:
      return s_ptr;
    case INTERP_MATRIX_ARRAY:
      return order_interp_ptr;
    case CUB2D_R:
      return cub_r_ptr;
    case CUB2D_S:
      return cub_s_ptr;
    case CUB2D_W:
      return cub_w_ptr;
    default:
      try {
        return dg_mats.at(matrix)->get_mat_ptr_dp();
      } catch (std::out_of_range &e) {
        dg_abort("This constant matrix is not supported by DGConstants2D\n");
      }
      return nullptr;
  }
}

// Get a pointer to the constant matrix in single precision (host memory).
// For matrices with multiple DG orders, the pointer will point to the 1st order matrix.
float* DGConstants2D::get_mat_ptr_sp(Constant_Matrix matrix) {
  switch(matrix) {
    case INTERP_MATRIX_ARRAY:
      return order_interp_ptr_sp;
    default:
      try {
        return dg_mats.at(matrix)->get_mat_ptr_sp();
      } catch (std::out_of_range &e) {
        dg_abort("This single precision constant matrix is not supported by DGConstants2D\n");
      }
      return nullptr;
  }
}

DGConstants2D::~DGConstants2D() {
  clean_up_kernel_ptrs();

  // Free all DGConstantMatrix objects
  for(auto &dg_mat : dg_mats) {
    delete dg_mat.second;
  }

  free(r_ptr);
  free(s_ptr);
  free(cub_r_ptr);
  free(cub_s_ptr);
  free(cub_w_ptr);
  free(order_interp_ptr);
  free(order_interp_ptr_sp);
}
