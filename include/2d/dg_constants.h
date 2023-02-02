#ifndef __DG_CONSTANTS_2D_H
#define __DG_CONSTANTS_2D_H

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "../dg_constants.h"

class DGConstants2D : public DGConstants {
public:
  DGConstants2D(const int n_);
  ~DGConstants2D();

  void calc_interp_mats() override;
  double* get_mat_ptr(Constant_Matrix matrix) override;

  int N_max, Nfp_max, Np_max, cNp_max, gNp_max, gNfp_max;

private:
  void cubature(const int nCub, const int N, arma::mat &V_, arma::mat &invV_, arma::mat &Dr_, arma::mat &Ds_);
  void gauss(const int nGauss, const int N, arma::mat &V_, arma::mat &invV_, arma::mat &Dr_, arma::mat &Ds_);

  // Pointers to all matrices that are returned by get_mat_ptr
  double *r_ptr, *s_ptr, *v_ptr, *invV_ptr, *mass_ptr, *invMass_ptr, *Dr_ptr;
  double *Ds_ptr, *Drw_ptr, *Dsw_ptr, *lift_ptr, *cubV_ptr, *cubDr_ptr;
  double *cubDs_ptr, *cubVDr_ptr, *cubVDs_ptr, *gInterp_ptr, *gFInterp0_ptr;
  double *gFInterp1_ptr, *gFInterp2_ptr, *gF0Dr_ptr, *gF0Ds_ptr, *gF1Dr_ptr, *gF1Ds_ptr;
  double *gF2Dr_ptr, *gF2Ds_ptr, *invMass_gInterpT_ptr;
  // Effectively a 2D array of interp matrices. Size [DG_ORDER][DG_ORDER]
  // To get an interp array:
  // int ind = ((order_from - 1) * DG_ORDER + (order_to - 1)) * DG_NP * DG_NP
  double *order_interp_ptr;
};

#endif
