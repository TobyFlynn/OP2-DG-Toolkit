#include "dg_constants/dg_constants_3d.h"

#include <stdexcept>

#include "dg_compiler_defs.h"
#include "dg_utils.h"

int FMASK[DG_ORDER * DG_NUM_FACES * DG_NPF];
int DG_CONSTANTS[DG_ORDER * DG_NUM_CONSTANTS];

int FMASK_TK[DG_ORDER * DG_NUM_FACES * DG_NPF];
int DG_CONSTANTS_TK[DG_ORDER * DG_NUM_CONSTANTS];

// TODO not require this
DG_FP cubW_g[1];
DG_FP gaussW_g[1];

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

DGConstants3D::DGConstants3D(const int n_) {
  // Set order
  N_max = n_;
  // Set num points and num face points
  DGUtils::numNodes3D(N_max, &Np_max, &Nfp_max);

  r_ptr   = (DG_FP *)calloc(N_max * Np_max, sizeof(DG_FP));
  s_ptr   = (DG_FP *)calloc(N_max * Np_max, sizeof(DG_FP));
  t_ptr   = (DG_FP *)calloc(N_max * Np_max, sizeof(DG_FP));
  Dr_ptr  = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  Ds_ptr  = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  Dt_ptr  = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  Drw_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  Dsw_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  Dtw_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  mass_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  invMass_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  invV_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  v_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  lift_ptr = (DG_FP *)calloc(N_max * DG_NUM_FACES * Nfp_max * Np_max, sizeof(DG_FP));
  mmF0_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  mmF1_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  mmF2_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  mmF3_ptr = (DG_FP *)calloc(N_max * Np_max * Np_max, sizeof(DG_FP));
  eMat_ptr = (DG_FP *)calloc(N_max * DG_NUM_FACES * Nfp_max * Np_max, sizeof(DG_FP));
  order_interp_ptr = (DG_FP *)calloc(N_max * N_max * Np_max * Np_max, sizeof(DG_FP));

  for(int N = 1; N <= N_max; N++) {
    int Np, Nfp;
    DGUtils::numNodes3D(N, &Np, &Nfp);

    arma::vec x_, y_, z_, r_, s_, t_;
    DGUtils::setRefXYZ(N, x_, y_, z_);
    DGUtils::xyz2rst(x_, y_, z_, r_, s_, t_);

    arma::mat v_    = DGUtils::vandermonde3D(r_, s_, t_, N);
    arma::mat invV_ = arma::inv(v_);
    arma::mat mass_ = invV_.t() * invV_;
    arma::mat inv_mass_ = arma::inv(mass_);
    arma::mat dr_, ds_, dt_;
    DGUtils::dMatrices3D(r_, s_, t_, v_, N, dr_, ds_, dt_);

    arma::uvec fmask1_ = arma::find(arma::abs(1 + t_)  < 1e-12);
    arma::uvec fmask2_ = arma::find(arma::abs(1 + s_) < 1e-12);
    arma::uvec fmask3_ = arma::find(arma::abs(1 + r_ + s_ + t_)  < 1e-12);
    arma::uvec fmask4_ = arma::find(arma::abs(1 + r_)  < 1e-12);
    arma::uvec fmask_  = arma::join_cols(fmask1_, fmask2_, fmask3_, fmask4_);

    arma::mat eMat_ = DGUtils::eMat3D(r_, s_, t_, fmask_, N);
    arma::mat lift_ = DGUtils::lift3D(r_, s_, t_, fmask_, v_, N);
    arma::mat mmF0_, mmF1_, mmF2_, mmF3_;
    DGUtils::faceMassMatrix3D(r_, s_, t_, fmask_, v_, N, mmF0_, mmF1_, mmF2_, mmF3_);

    arma::mat vr, vs, vt;
    DGUtils::gradVandermonde3D(r_, s_, t_, N, vr, vs, vt);
    arma::mat drw_ = (v_ * vr.t()) * arma::inv(v_ * v_.t());
    arma::mat dsw_ = (v_ * vs.t()) * arma::inv(v_ * v_.t());
    arma::mat dtw_ = (v_ * vt.t()) * arma::inv(v_ * v_.t());

    save_vec(r_ptr, r_, N, Np_max);
    save_vec(s_ptr, s_, N, Np_max);
    save_vec(t_ptr, t_, N, Np_max);
    save_mat(Dr_ptr, dr_, N, Np_max * Np_max);
    save_mat(Ds_ptr, ds_, N, Np_max * Np_max);
    save_mat(Dt_ptr, dt_, N, Np_max * Np_max);
    save_mat(Drw_ptr, drw_, N, Np_max * Np_max);
    save_mat(Dsw_ptr, dsw_, N, Np_max * Np_max);
    save_mat(Dtw_ptr, dtw_, N, Np_max * Np_max);
    save_mat(mass_ptr, mass_, N, Np_max * Np_max);
    save_mat(invMass_ptr, inv_mass_, N, Np_max * Np_max);
    save_mat(invV_ptr, invV_, N, Np_max * Np_max);
    save_mat(v_ptr, v_, N, Np_max * Np_max);
    save_mat(lift_ptr, lift_, N, DG_NUM_FACES * Nfp_max * Np_max);
    save_mat(mmF0_ptr, mmF0_, N, Np_max * Np_max);
    save_mat(mmF1_ptr, mmF1_, N, Np_max * Np_max);
    save_mat(mmF2_ptr, mmF2_, N, Np_max * Np_max);
    save_mat(mmF3_ptr, mmF3_, N, Np_max * Np_max);
    save_mat(eMat_ptr, eMat_, N, DG_NUM_FACES * Nfp_max * Np_max);

    std::vector<int> fmask_int = arma::conv_to<std::vector<int>>::from(fmask_);
    memcpy(&FMASK[(N - 1) * DG_NUM_FACES * Nfp_max], fmask_int.data(), fmask_int.size() * sizeof(int));
    memcpy(&FMASK_TK[(N - 1) * DG_NUM_FACES * Nfp_max], fmask_int.data(), fmask_int.size() * sizeof(int));

    DG_CONSTANTS[(N - 1) * DG_NUM_CONSTANTS]     = Np;
    DG_CONSTANTS[(N - 1) * DG_NUM_CONSTANTS + 1] = Nfp;
    DG_CONSTANTS_TK[(N - 1) * DG_NUM_CONSTANTS]     = Np;
    DG_CONSTANTS_TK[(N - 1) * DG_NUM_CONSTANTS + 1] = Nfp;
  }
}

void DGConstants3D::calc_interp_mats() {
  // From n0 to n1
  for(int n0 = 1; n0 <= N_max; n0++) {
    arma::vec x_0, y_0, z_0, r_0, s_0, t_0;
    DGUtils::setRefXYZ(n0, x_0, y_0, z_0);
    DGUtils::xyz2rst(x_0, y_0, z_0, r_0, s_0, t_0);
    arma::mat v_0    = DGUtils::vandermonde3D(r_0, s_0, t_0, n0);
    arma::mat invV_0 = arma::inv(v_0);
    for(int n1 = 1; n1 <= N_max; n1++) {
      if(n0 != n1) {
        arma::vec x_1, y_1, z_1, r_1, s_1, t_1;
        DGUtils::setRefXYZ(n1, x_1, y_1, z_1);
        DGUtils::xyz2rst(x_1, y_1, z_1, r_1, s_1, t_1);
        arma::mat v_1    = DGUtils::vandermonde3D(r_1, s_1, t_1, n1);
        arma::mat invV_1 = arma::inv(v_1);
        arma::mat interp_;
        // arma::mat interp_ = DGUtils::interpMatrix3D(r_1, s_1, t_1, invV_0, n0);
        if(n1 > n0) {
          // interp_ = DGUtils::interpMatrix3D(r_0, s_0, t_0, invV_1, n1).t();
          // interp_ = DGUtils::interpMatrix3D(r_1, s_1, t_1, invV_0, n0);
          interp_ = DGUtils::interpMatrix3D(r_1, s_1, t_1, invV_0, n0);

          // arma::mat tmp_M(r_0.n_elem, r_0.n_elem, arma::fill::zeros);
          // arma::mat tmp_vT_inv_1 = arma::inv(v_1.t());
          // arma::mat tmp_vT_inv_0 = arma::inv(v_0.t());
          // for(int i = 0; i < r_0.n_elem; i++) {
          //   for(int j = 0; j < r_0.n_elem; j++) {
          //     for(int n = 0; n < r_0.n_elem; n++) {
          //       tmp_M(i, j) += tmp_vT_inv_0(i, n) * tmp_vT_inv_0(j, n);
          //     }
          //   }
          // }
          // arma::mat interp_points = DGUtils::interpMatrix3D(r_1, s_1, t_1, invV_0, n0);
          // interp_ = arma::inv(invV_1.t() * invV_1) * interp_points * tmp_M;
        } else {
          // interp_ = DGUtils::interpMatrix3D(r_1, s_1, t_1, invV_0, n0);
          // interp_ = DGUtils::interpMatrix3D(r_0, s_0, t_0, invV_1, n1).t();
          interp_ = DGUtils::interpMatrix3D(r_0, s_0, t_0, invV_1, n1).t();

          // arma::mat tmp_M(r_1.n_elem, r_1.n_elem, arma::fill::zeros);
          // arma::mat tmp_vT_inv_1 = arma::inv(v_1.t());
          // arma::mat tmp_vT_inv_0 = arma::inv(v_0.t());
          // for(int i = 0; i < r_1.n_elem; i++) {
          //   for(int j = 0; j < r_1.n_elem; j++) {
          //     for(int n = 0; n < r_0.n_elem; n++) {
          //       tmp_M(i, j) += tmp_vT_inv_1(i, n) * tmp_vT_inv_1(j, n);
          //     }
          //   }
          // }
          // arma::mat interp_points = DGUtils::interpMatrix3D(r_0, s_0, t_0, invV_1, n1);
          // interp_ = arma::inv(invV_0.t() * invV_0) * interp_points * tmp_M;
          // interp_ = interp_.t();
        }
        #ifdef DG_COL_MAJ
        arma::Mat<DG_FP> interp_2 = arma::conv_to<arma::Mat<DG_FP>>::from(interp_);
        #else
        arma::Mat<DG_FP> interp_2 = arma::conv_to<arma::Mat<DG_FP>>::from(interp_.t());
        #endif
        memcpy(&order_interp_ptr[((n0 - 1) * N_max + (n1 - 1)) * Np_max * Np_max], interp_2.memptr(), interp_2.n_elem * sizeof(DG_FP));
      }
    }
  }


  // for(int p0 = 0; p0 < N_max; p0++) {
  //   for(int p1 = p0 + 1; p1 < N_max; p1++) {
  //     memcpy(&order_interp_ptr[(p1 * DG_ORDER + p0) * DG_NP * DG_NP], &order_interp_ptr[(p0 * DG_ORDER + p1) * DG_NP * DG_NP], DG_NP * DG_NP * sizeof(DG_FP));
  //   }
  // }

  transfer_kernel_ptrs();
}

DG_FP* DGConstants3D::get_mat_ptr(Constant_Matrix matrix) {
  switch(matrix) {
    case R:
      return r_ptr;
    case S:
      return s_ptr;
    case T:
      return t_ptr;
    case DR:
      return Dr_ptr;
    case DS:
      return Ds_ptr;
    case DT:
      return Dt_ptr;
    case DRW:
      return Drw_ptr;
    case DSW:
      return Dsw_ptr;
    case DTW:
      return Dtw_ptr;
    case MASS:
      return mass_ptr;
    case INV_MASS:
      return invMass_ptr;
    case INV_V:
      return invV_ptr;
    case V:
      return v_ptr;
    case LIFT:
      return lift_ptr;
    case MM_F0:
      return mmF0_ptr;
    case MM_F1:
      return mmF1_ptr;
    case MM_F2:
      return mmF2_ptr;
    case MM_F3:
      return mmF3_ptr;
    case EMAT:
      return eMat_ptr;
    case INTERP_MATRIX_ARRAY:
      return order_interp_ptr;
    default:
      throw std::runtime_error("This constant matrix is not supported by DGConstants3D\n");
      return nullptr;
  }
}

DGConstants3D::~DGConstants3D() {
  clean_up_kernel_ptrs();

  delete r_ptr;
  delete s_ptr;
  delete t_ptr;
  delete Dr_ptr;
  delete Ds_ptr;
  delete Dt_ptr;
  delete Drw_ptr;
  delete Dsw_ptr;
  delete Dtw_ptr;
  delete mass_ptr;
  delete invMass_ptr;
  delete invV_ptr;
  delete v_ptr;
  delete lift_ptr;
  delete mmF0_ptr;
  delete mmF1_ptr;
  delete mmF2_ptr;
  delete mmF3_ptr;
  delete eMat_ptr;
  delete order_interp_ptr;
}
