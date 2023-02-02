#ifndef __DG_CONSTANTS_3D_H
#define __DG_CONSTANTS_3D_H

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "../dg_constants.h"

class DGConstants3D : public DGConstants {
public:
  DGConstants3D(const int n_);
  ~DGConstants3D();

  void calc_interp_mats() override;
  double* get_mat_ptr(Constant_Matrix matrix) override;

  int N_max, Np_max, Nfp_max;

private:
  double *Dr_ptr, *Ds_ptr, *Dt_ptr, *Drw_ptr, *Dsw_ptr, *Dtw_ptr;
  double *mass_ptr, *invMass_ptr, *invV_ptr, *lift_ptr;
  double *mmF0_ptr, *mmF1_ptr, *mmF2_ptr, *mmF3_ptr;
  // Effectively a 2D array of interp matrices. Size [DG_ORDER][DG_ORDER]
  // To get an interp array:
  // int ind = ((order_from - 1) * DG_ORDER + (order_to - 1)) * DG_NP * DG_NP
  double *order_interp_ptr;
};

#endif