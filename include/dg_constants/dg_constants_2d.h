#ifndef __DG_CONSTANTS_2D_H
#define __DG_CONSTANTS_2D_H

#include "dg_compiler_defs.h"

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "dg_constants.h"

class DGConstants2D : public DGConstants {
public:
  DGConstants2D(const int n_);
  ~DGConstants2D();

  void calc_interp_mats() override;
  DG_FP* get_mat_ptr(Constant_Matrix matrix) override;
  DG_FP* get_mat_ptr_device(Constant_Matrix matrix) override;
  float* get_mat_ptr_device_sp(Constant_Matrix matrix) override;

  int cNp_max, gNp_max, gNfp_max;

private:
  void cubature(const int nCub);
  void gauss(const int nGauss);

  void transfer_kernel_ptrs();
  void clean_up_kernel_ptrs();

  // Pointers to all matrices that are returned by get_mat_ptr
  DG_FP *r_ptr, *s_ptr, *v_ptr, *invV_ptr, *mass_ptr, *invMass_ptr, *Dr_ptr;
  DG_FP *Ds_ptr, *Drw_ptr, *Dsw_ptr, *lift_ptr;
  DG_FP *cub_r_ptr, *cub_s_ptr, *cub_w_ptr;
  DG_FP *cubInterp_ptr, *cubProj_ptr, *cubPDrT_ptr, *cubPDsT_ptr;
  DG_FP *cubInterpSurf_ptr, *cubLiftSurf_ptr;
  DG_FP *mmF0_ptr, *mmF1_ptr, *mmF2_ptr, *eMat_ptr;
  // Effectively a 2D array of interp matrices. Size [DG_ORDER][DG_ORDER]
  // To get an interp array:
  // int ind = ((order_from - 1) * DG_ORDER + (order_to - 1)) * DG_NP * DG_NP
  DG_FP *order_interp_ptr;

  float *Dr_ptr_sp, *Ds_ptr_sp, *Drw_ptr_sp, *Dsw_ptr_sp, *mass_ptr_sp, *invMass_ptr_sp;
  float *invV_ptr_sp, *v_ptr_sp, *lift_ptr_sp, *eMat_ptr_sp, *order_interp_ptr_sp;
};

#endif
