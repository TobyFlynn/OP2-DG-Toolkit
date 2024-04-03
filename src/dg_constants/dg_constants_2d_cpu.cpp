#include "dg_constants/dg_constants_2d.h"

#include "dg_abort.h"

DG_FP *dg_r_kernel;
DG_FP *dg_s_kernel;
DG_FP *dg_Dr_kernel;
DG_FP *dg_Ds_kernel;
DG_FP *dg_Drw_kernel;
DG_FP *dg_Dsw_kernel;
DG_FP *dg_Mass_kernel;
DG_FP *dg_InvMass_kernel;
DG_FP *dg_InvV_kernel;
DG_FP *dg_V_kernel;
DG_FP *dg_Lift_kernel;
DG_FP *dg_Interp_kernel;
DG_FP *dg_MM_F0_kernel;
DG_FP *dg_MM_F1_kernel;
DG_FP *dg_MM_F2_kernel;
DG_FP *dg_Emat_kernel;
DG_FP *dg_cubSurf2d_Interp_kernel;

// Set up pointers that are accessible within OP2 kernels
void DGConstants2D::transfer_kernel_ptrs() {
  dg_r_kernel = r_ptr;
  dg_s_kernel = s_ptr;
  dg_Dr_kernel = dg_mats.at(DR)->get_mat_ptr_dp();
  dg_Ds_kernel = dg_mats.at(DS)->get_mat_ptr_dp();
  dg_Drw_kernel = dg_mats.at(DRW)->get_mat_ptr_dp();
  dg_Dsw_kernel = dg_mats.at(DSW)->get_mat_ptr_dp();
  dg_Mass_kernel = dg_mats.at(MASS)->get_mat_ptr_dp();
  dg_InvMass_kernel = dg_mats.at(INV_MASS)->get_mat_ptr_dp();
  dg_InvV_kernel = dg_mats.at(INV_V)->get_mat_ptr_dp();
  dg_V_kernel = dg_mats.at(V)->get_mat_ptr_dp();
  dg_Lift_kernel = dg_mats.at(LIFT)->get_mat_ptr_dp();
  dg_Interp_kernel = order_interp_ptr;
  dg_MM_F0_kernel = dg_mats.at(MM_F0)->get_mat_ptr_dp();
  dg_MM_F1_kernel = dg_mats.at(MM_F1)->get_mat_ptr_dp();
  dg_MM_F2_kernel = dg_mats.at(MM_F2)->get_mat_ptr_dp();
  dg_Emat_kernel = dg_mats.at(EMAT)->get_mat_ptr_dp();
  dg_cubSurf2d_Interp_kernel = dg_mats.at(CUBSURF2D_INTERP)->get_mat_ptr_dp();
}

void DGConstants2D::clean_up_kernel_ptrs() {
  // Do nothing for CPU
}

DG_FP* DGConstants2D::get_mat_ptr_device(Constant_Matrix matrix) {
  switch(matrix) {
    case R:
      return dg_r_kernel;
    case S:
      return dg_s_kernel;
    case INTERP_MATRIX_ARRAY:
      return dg_Interp_kernel;
    default:
      try {
        return dg_mats.at(matrix)->get_mat_ptr_dp_device();
      } catch (std::out_of_range &e) {
        dg_abort("This double-precision constant matrix is not supported by DGConstants2D\n");
      }
      return nullptr;
  }
}

float* DGConstants2D::get_mat_ptr_device_sp(Constant_Matrix matrix) {
 switch(matrix) {
    case INTERP_MATRIX_ARRAY:
      return order_interp_ptr_sp;
    default:
      try {
          return dg_mats.at(matrix)->get_mat_ptr_sp_device();
        } catch (std::out_of_range &e) {
          dg_abort("This single-precision constant matrix is not supported by DGConstants2D\n");
        }
      return nullptr;
  }
}
