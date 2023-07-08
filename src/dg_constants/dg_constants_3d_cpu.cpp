#include "dg_constants/dg_constants_3d.h"

#include <stdexcept>

DG_FP *dg_r_kernel;
DG_FP *dg_s_kernel;
DG_FP *dg_t_kernel;
DG_FP *dg_Dr_kernel;
DG_FP *dg_Ds_kernel;
DG_FP *dg_Dt_kernel;
DG_FP *dg_Drw_kernel;
DG_FP *dg_Dsw_kernel;
DG_FP *dg_Dtw_kernel;
DG_FP *dg_Mass_kernel;
DG_FP *dg_InvMass_kernel;
DG_FP *dg_InvV_kernel;
DG_FP *dg_V_kernel;
DG_FP *dg_Lift_kernel;
DG_FP *dg_MM_F0_kernel;
DG_FP *dg_MM_F1_kernel;
DG_FP *dg_MM_F2_kernel;
DG_FP *dg_MM_F3_kernel;
DG_FP *dg_Emat_kernel;
DG_FP *dg_Interp_kernel;

void DGConstants3D::transfer_kernel_ptrs() {
  dg_r_kernel = r_ptr;
  dg_s_kernel = s_ptr;
  dg_t_kernel = t_ptr;
  dg_Dr_kernel = Dr_ptr;
  dg_Ds_kernel = Ds_ptr;
  dg_Dt_kernel = Dt_ptr;
  dg_Drw_kernel = Drw_ptr;
  dg_Dsw_kernel = Dsw_ptr;
  dg_Dtw_kernel = Dtw_ptr;
  dg_Mass_kernel = mass_ptr;
  dg_InvMass_kernel = invMass_ptr;
  dg_InvV_kernel = invV_ptr;
  dg_V_kernel = v_ptr;
  dg_Lift_kernel = lift_ptr;
  dg_MM_F0_kernel = mmF0_ptr;
  dg_MM_F1_kernel = mmF1_ptr;
  dg_MM_F2_kernel = mmF2_ptr;
  dg_MM_F3_kernel = mmF3_ptr;
  dg_Emat_kernel = eMat_ptr;
  dg_Interp_kernel = order_interp_ptr;
}

void DGConstants3D::clean_up_kernel_ptrs() {
  // Do nothing for CPU
}

DG_FP* DGConstants3D::get_mat_ptr_kernel(Constant_Matrix matrix) {
  switch(matrix) {
    case R:
      return dg_r_kernel;
    case S:
      return dg_s_kernel;
    case T:
      return dg_t_kernel;
    case DR:
      return dg_Dr_kernel;
    case DS:
      return dg_Ds_kernel;
    case DT:
      return dg_Dt_kernel;
    case DRW:
      return dg_Drw_kernel;
    case DSW:
      return dg_Dsw_kernel;
    case DTW:
      return dg_Dtw_kernel;
    case MASS:
      return dg_Mass_kernel;
    case INV_MASS:
      return dg_InvMass_kernel;
    case INV_V:
      return dg_InvV_kernel;
    case V:
      return dg_V_kernel;
    case LIFT:
      return dg_Lift_kernel;
    case MM_F0:
      return dg_MM_F0_kernel;
    case MM_F1:
      return dg_MM_F1_kernel;
    case MM_F2:
      return dg_MM_F2_kernel;
    case MM_F3:
      return dg_MM_F3_kernel;
    case EMAT:
      return dg_Emat_kernel;
    case INTERP_MATRIX_ARRAY:
      return dg_Interp_kernel;
    default:
      throw std::runtime_error("This constant matrix is not supported by DGConstants3D\n");
      return nullptr;
  }
}
