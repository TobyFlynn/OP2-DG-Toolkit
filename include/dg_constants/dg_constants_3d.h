#ifndef __DG_CONSTANTS_3D_H
#define __DG_CONSTANTS_3D_H

#include "dg_compiler_defs.h"

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "dg_constants.h"

class DGConstants3D : public DGConstants {
public:
  DGConstants3D(const int n_);
  ~DGConstants3D();

  void calc_interp_mats() override;
  DG_FP* get_mat_ptr(Constant_Matrix matrix) override;
  DG_FP* get_mat_ptr_kernel(Constant_Matrix matrix) override;
  float* get_mat_ptr_kernel_sp(Constant_Matrix matrix) override;

private:
  void getCubatureData(const int N, arma::vec &cubr, arma::vec &cubs, arma::vec &cubt, arma::vec &cubw);
  void transfer_kernel_ptrs();
  void clean_up_kernel_ptrs();

  DG_FP *r_ptr, *s_ptr, *t_ptr;
  DG_FP *Dr_ptr, *Ds_ptr, *Dt_ptr, *Drw_ptr, *Dsw_ptr, *Dtw_ptr;
  DG_FP *mass_ptr, *invMass_ptr, *invV_ptr, *v_ptr, *lift_ptr;
  DG_FP *mmF0_ptr, *mmF1_ptr, *mmF2_ptr, *mmF3_ptr, *eMat_ptr;
  // Effectively a 2D array of interp matrices. Size [DG_ORDER][DG_ORDER]
  // To get an interp array:
  // int ind = ((order_from - 1) * DG_ORDER + (order_to - 1)) * DG_NP * DG_NP
  DG_FP *order_interp_ptr;
  DG_FP *cub_r_ptr, *cub_s_ptr, *cub_t_ptr, *cub_w_ptr;
  DG_FP *cubInterp_ptr, *cubProj_ptr, *cubPDrT_ptr, *cubPDsT_ptr, *cubPDtT_ptr;
  DG_FP *cubInterpSurf_ptr, *cubLiftSurf_ptr;

  float *Dr_ptr_sp, *Ds_ptr_sp, *Dt_ptr_sp, *Drw_ptr_sp, *Dsw_ptr_sp, *Dtw_ptr_sp;
  float *mass_ptr_sp, *invMass_ptr_sp, *invV_ptr_sp, *v_ptr_sp, *lift_ptr_sp, *eMat_ptr_sp;
  float *order_interp_ptr_sp;
};

#endif
