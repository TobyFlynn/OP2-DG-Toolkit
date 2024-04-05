#include "dg_constants/dg_constants_3d.h"

#include "dg_abort.h"

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
DG_FP *dg_cub3d_Interp_kernel;
DG_FP *dg_cub3d_Proj_kernel;
DG_FP *dg_cub3d_PDr_kernel;
DG_FP *dg_cub3d_PDs_kernel;
DG_FP *dg_cub3d_PDt_kernel;
DG_FP *dg_cubSurf3d_Interp_kernel;
DG_FP *dg_cubSurf3d_Lift_kernel;

void DGConstants3D::transfer_kernel_ptrs() {
  dg_r_kernel = get_mat_ptr(R);
  dg_s_kernel = get_mat_ptr(S);
  dg_t_kernel = get_mat_ptr(T);
  dg_Dr_kernel = get_mat_ptr(DR);
  dg_Ds_kernel = get_mat_ptr(DS);
  dg_Dt_kernel = get_mat_ptr(DT);
  dg_Drw_kernel = get_mat_ptr(DRW);
  dg_Dsw_kernel = get_mat_ptr(DSW);
  dg_Dtw_kernel = get_mat_ptr(DTW);
  dg_Mass_kernel = get_mat_ptr(MASS);
  dg_InvMass_kernel = get_mat_ptr(INV_MASS);
  dg_InvV_kernel = get_mat_ptr(INV_V);
  dg_V_kernel = get_mat_ptr(V);
  dg_Lift_kernel = get_mat_ptr(LIFT);
  dg_MM_F0_kernel = get_mat_ptr(MM_F0);
  dg_MM_F1_kernel = get_mat_ptr(MM_F1);
  dg_MM_F2_kernel = get_mat_ptr(MM_F2);
  dg_MM_F3_kernel = get_mat_ptr(MM_F3);
  dg_Emat_kernel = get_mat_ptr(EMAT);
  dg_Interp_kernel = get_mat_ptr(INTERP_MATRIX_ARRAY);
  dg_cub3d_Interp_kernel = get_mat_ptr(CUB3D_INTERP);
  dg_cub3d_Proj_kernel = get_mat_ptr(CUB3D_PROJ);
  dg_cub3d_PDr_kernel = get_mat_ptr(CUB3D_PDR);
  dg_cub3d_PDs_kernel = get_mat_ptr(CUB3D_PDS);
  dg_cub3d_PDt_kernel = get_mat_ptr(CUB3D_PDT);
  dg_cubSurf3d_Interp_kernel = get_mat_ptr(CUBSURF3D_INTERP);
  dg_cubSurf3d_Lift_kernel = get_mat_ptr(CUBSURF3D_LIFT);
}

void DGConstants3D::clean_up_kernel_ptrs() {
  // Do nothing for CPU
}

DG_FP* DGConstants3D::get_mat_ptr_device(Constant_Matrix matrix) {
  return get_mat_ptr(matrix);
}

float* DGConstants3D::get_mat_ptr_device_sp(Constant_Matrix matrix) {
  return get_mat_ptr_sp(matrix);
}
