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

private:
  void transfer_kernel_ptrs();
  void clean_up_kernel_ptrs();

  DG_FP *r_ptr, *s_ptr, *t_ptr;
  DG_FP *Dr_ptr, *Ds_ptr, *Dt_ptr, *Drw_ptr, *Dsw_ptr, *Dtw_ptr;
  DG_FP *mass_ptr, *invMass_ptr, *invV_ptr, *lift_ptr;
  DG_FP *mmF0_ptr, *mmF1_ptr, *mmF2_ptr, *mmF3_ptr, *eMat_ptr;
  // Effectively a 2D array of interp matrices. Size [DG_ORDER][DG_ORDER]
  // To get an interp array:
  // int ind = ((order_from - 1) * DG_ORDER + (order_to - 1)) * DG_NP * DG_NP
  DG_FP *order_interp_ptr;
};

#endif
