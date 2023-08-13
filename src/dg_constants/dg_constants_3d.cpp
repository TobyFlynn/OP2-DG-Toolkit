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

  printf("pre pre\n");

  // 3D volume cubature stuff
  int Np, Nfp;
  DGUtils::numNodes3D(DG_ORDER, &Np, &Nfp);
  arma::vec x_, y_, z_, r_, s_, t_;
  DGUtils::setRefXYZ(DG_ORDER, x_, y_, z_);
  DGUtils::xyz2rst(x_, y_, z_, r_, s_, t_);
  arma::mat v_    = DGUtils::vandermonde3D(r_, s_, t_, DG_ORDER);
  arma::mat invV_ = arma::inv(v_);

  arma::vec cubr, cubs, cubt, cubw;
  getCubatureData(cubr, cubs, cubt, cubw);
  printf("pre\n");
  arma::mat cubInterp = DGUtils::interpMatrix3D(cubr, cubs, cubt, invV_, DG_ORDER); printf("0\n");
  arma::mat cubProj   = DGUtils::cubaturePMat3D(r_, s_, t_, cubr, cubs, cubt, DG_ORDER); printf("1\n");
  arma::mat cubPDrT, cubPDsT, cubPDtT;
  DGUtils::cubaturePDwMat3D(r_, s_, t_, cubr, cubs, cubt, DG_ORDER, cubPDrT, cubPDsT, cubPDtT); printf("2\n");
  const int cubNp = cubr.n_elem;
  arma::mat diag_w(cubNp, cubNp);
  diag_w.zeros();
  for(int i = 0; i < cubNp; i++) {
    diag_w(i,i) = cubw(i);
  }
  printf("3\n");
  cubProj = cubProj * diag_w;
  printf("3.5\n");
  cubPDrT = cubPDrT * diag_w;
  cubPDsT = cubPDsT * diag_w;
  cubPDtT = cubPDtT * diag_w;
  printf("4\n");
  printf("%dx%d\n", cubInterp.n_rows, cubInterp.n_cols);
  printf("%dx%d\n", cubProj.n_rows, cubProj.n_cols);
  printf("%dx%d\n", cubPDrT.n_rows, cubPDrT.n_cols);

  cubInterp_ptr = (DG_FP *)calloc(DG_NP * DG_CUB_3D_NP, sizeof(DG_FP));
  cubProj_ptr = (DG_FP *)calloc(DG_NP * DG_CUB_3D_NP, sizeof(DG_FP));
  cubPDrT_ptr = (DG_FP *)calloc(DG_NP * DG_CUB_3D_NP, sizeof(DG_FP));
  cubPDsT_ptr = (DG_FP *)calloc(DG_NP * DG_CUB_3D_NP, sizeof(DG_FP));
  cubPDtT_ptr = (DG_FP *)calloc(DG_NP * DG_CUB_3D_NP, sizeof(DG_FP));

  save_mat(cubInterp_ptr, cubInterp, 1, DG_NP * DG_CUB_3D_NP);
  save_mat(cubProj_ptr, cubProj, 1, DG_NP * DG_CUB_3D_NP);
  save_mat(cubPDrT_ptr, cubPDrT, 1, DG_NP * DG_CUB_3D_NP);
  save_mat(cubPDsT_ptr, cubPDsT, 1, DG_NP * DG_CUB_3D_NP);
  save_mat(cubPDtT_ptr, cubPDtT, 1, DG_NP * DG_CUB_3D_NP);
}

void DGConstants3D::getCubatureData(arma::vec &cubr, arma::vec &cubs, arma::vec &cubt, arma::vec &cubw) {
  const double cubR6[23] = {-9.223278313102310e-01,-8.704611261398938e-01,-8.704496791057892e-01,-4.441926613398445e-01,-8.678026751706386e-01,-3.497606828459493e-01,-3.616114393021378e-01,-3.432236575375570e-01,-8.898019550185483e-01,-7.507000727250277e-01,-8.681501536799804e-01,-9.852909523238611e-01, 2.349114402945376e-01,-4.411598941080236e-01,-4.245498103470715e-01, 1.894346037515911e-01,-8.664208004365244e-01, 2.530804034177644e-01,-8.799788339594619e-01,-4.484273990602982e-01,-8.973495876695942e-01,-9.188478978663642e-01, 8.075400026643635e-01};
  const double cubS6[23] = {-9.513620515037141e-01,-4.643116036328488e-01,-9.530644088538914e-01,-8.725342094100046e-01,-8.326423718798897e-01,-3.412405629016029e-01,-3.916614693004364e-01,-9.234226585235092e-01,-2.961216053305913e-01,-6.957923773801387e-01, 2.486427271068594e-01,-5.774046828368271e-01,-8.736000381148603e-01,-4.883584314700274e-01, 1.546915627794538e-01,-8.696440144732590e-01, 6.012655096203365e-02,-5.031009197622103e-01,-5.739176335276283e-01,-8.920077183281708e-01, 6.822779033246367e-01,-9.824360844449622e-01,-9.542683523719533e-01};
  const double cubT6[23] = { 8.058575980272226e-01, 2.735350171170275e-01,-2.182758986579757e-01, 1.898193780435916e-01, 2.601091102219790e-01,-3.463329907619084e-01,-9.112333112855833e-01,-3.594251326046153e-01,-2.378313821873799e-01,-5.975308652711581e-01,-4.928126505135995e-01,-4.976310094449407e-01,-4.831017020321489e-01,-4.608601407334560e-01,-8.707587238532627e-01,-8.667934039847933e-01,-8.460145657980651e-01,-8.757689336328026e-01,-9.483146274785937e-01,-8.799677016676628e-01,-9.254704957232888e-01,-8.227992990621793e-01,-9.413285578336420e-01};
  const double cubW6[23] = { 9.461059802212705e-03, 4.201254651027517e-02, 3.230838250325908e-02, 6.518676786992283e-02, 7.071109854422583e-02, 5.765239559396446e-02, 8.951442161674263e-02, 7.976179688190549e-02, 8.348596704174839e-02, 8.577869596411661e-02, 5.812053074750553e-02, 3.008755637085715e-02, 6.215084550210757e-02, 1.501325393254235e-01, 7.191334750441594e-02, 6.635817345535235e-02, 8.408848251402726e-02, 6.286404062968159e-02, 3.400576568938985e-02, 5.295213019877631e-02, 2.123397224667169e-02, 1.389778096492792e-02, 9.655035855822626e-03};

  arma::vec r_tmp(cubR6, 23);
  arma::vec s_tmp(cubS6, 23);
  arma::vec t_tmp(cubT6, 23);
  arma::vec w_tmp(cubW6, 23);

  cubr = r_tmp;
  cubs = s_tmp;
  cubt = t_tmp;
  cubw = w_tmp;
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

  Dr_ptr_sp = (float *)calloc(N_max * Np_max * Np_max, sizeof(float));
  Ds_ptr_sp = (float *)calloc(N_max * Np_max * Np_max, sizeof(float));
  Dt_ptr_sp = (float *)calloc(N_max * Np_max * Np_max, sizeof(float));
  Drw_ptr_sp = (float *)calloc(N_max * Np_max * Np_max, sizeof(float));
  Dsw_ptr_sp = (float *)calloc(N_max * Np_max * Np_max, sizeof(float));
  Dtw_ptr_sp = (float *)calloc(N_max * Np_max * Np_max, sizeof(float));
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
    Dt_ptr_sp[i] = (float)Dt_ptr[i];
    Drw_ptr_sp[i] = (float)Drw_ptr[i];
    Dsw_ptr_sp[i] = (float)Dsw_ptr[i];
    Dtw_ptr_sp[i] = (float)Dtw_ptr[i];
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
    case CUB3D_INTERP:
      return cubInterp_ptr;
    case CUB3D_PROJ:
      return cubProj_ptr;
    case CUB3D_PDR:
      return cubPDrT_ptr;
    case CUB3D_PDS:
      return cubPDsT_ptr;
    case CUB3D_PDT:
      return cubPDtT_ptr;
    default:
      throw std::runtime_error("This constant matrix is not supported by DGConstants3D\n");
      return nullptr;
  }
}

DGConstants3D::~DGConstants3D() {
  clean_up_kernel_ptrs();

  free(r_ptr);
  free(s_ptr);
  free(t_ptr);
  free(Dr_ptr);
  free(Ds_ptr);
  free(Dt_ptr);
  free(Drw_ptr);
  free(Dsw_ptr);
  free(Dtw_ptr);
  free(mass_ptr);
  free(invMass_ptr);
  free(invV_ptr);
  free(v_ptr);
  free(lift_ptr);
  free(mmF0_ptr);
  free(mmF1_ptr);
  free(mmF2_ptr);
  free(mmF3_ptr);
  free(eMat_ptr);
  free(order_interp_ptr);
  free(cubInterp_ptr);
  free(cubProj_ptr);
  free(cubPDrT_ptr);
  free(cubPDsT_ptr);
  free(cubPDtT_ptr);

  free(Dr_ptr_sp);
  free(Ds_ptr_sp);
  free(Dt_ptr_sp);
  free(Drw_ptr_sp);
  free(Dsw_ptr_sp);
  free(Dtw_ptr_sp);
  free(mass_ptr_sp);
  free(invMass_ptr_sp);
  free(invV_ptr_sp);
  free(v_ptr_sp);
  free(lift_ptr_sp);
  free(eMat_ptr_sp);
  free(order_interp_ptr_sp);
}
