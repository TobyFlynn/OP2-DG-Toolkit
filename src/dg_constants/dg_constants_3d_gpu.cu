#include "dg_constants/dg_constants_3d.h"

#include "op_cuda_rt_support.h"

#include <stdexcept>

__constant__ DG_FP *dg_r_kernel;
__constant__ DG_FP *dg_s_kernel;
__constant__ DG_FP *dg_t_kernel;
__constant__ DG_FP *dg_Dr_kernel;
__constant__ DG_FP *dg_Ds_kernel;
__constant__ DG_FP *dg_Dt_kernel;
__constant__ DG_FP *dg_Drw_kernel;
__constant__ DG_FP *dg_Dsw_kernel;
__constant__ DG_FP *dg_Dtw_kernel;
__constant__ DG_FP *dg_Mass_kernel;
__constant__ DG_FP *dg_InvMass_kernel;
__constant__ DG_FP *dg_InvV_kernel;
__constant__ DG_FP *dg_V_kernel;
__constant__ DG_FP *dg_Lift_kernel;
__constant__ DG_FP *dg_MM_F0_kernel;
__constant__ DG_FP *dg_MM_F1_kernel;
__constant__ DG_FP *dg_MM_F2_kernel;
__constant__ DG_FP *dg_MM_F3_kernel;
__constant__ DG_FP *dg_Emat_kernel;
__constant__ DG_FP *dg_Interp_kernel;
__constant__ DG_FP *dg_cub3d_Interp_kernel;
__constant__ DG_FP *dg_cub3d_Proj_kernel;
__constant__ DG_FP *dg_cub3d_PDr_kernel;
__constant__ DG_FP *dg_cub3d_PDs_kernel;
__constant__ DG_FP *dg_cub3d_PDt_kernel;

DG_FP *dg_r_d;
DG_FP *dg_s_d;
DG_FP *dg_t_d;
DG_FP *dg_Dr_d;
DG_FP *dg_Ds_d;
DG_FP *dg_Dt_d;
DG_FP *dg_Drw_d;
DG_FP *dg_Dsw_d;
DG_FP *dg_Dtw_d;
DG_FP *dg_Mass_d;
DG_FP *dg_InvMass_d;
DG_FP *dg_InvV_d;
DG_FP *dg_V_d;
DG_FP *dg_Lift_d;
DG_FP *dg_MM_F0_d;
DG_FP *dg_MM_F1_d;
DG_FP *dg_MM_F2_d;
DG_FP *dg_MM_F3_d;
DG_FP *dg_Emat_d;
DG_FP *dg_Interp_d;
DG_FP *dg_cub3d_Interp_d;
DG_FP *dg_cub3d_Proj_d;
DG_FP *dg_cub3d_PDr_d;
DG_FP *dg_cub3d_PDs_d;
DG_FP *dg_cub3d_PDt_d;

float *dg_Dr_sp_d;
float *dg_Ds_sp_d;
float *dg_Dt_sp_d;
float *dg_Drw_sp_d;
float *dg_Dsw_sp_d;
float *dg_Dtw_sp_d;
float *dg_Mass_sp_d;
float *dg_InvMass_sp_d;
float *dg_InvV_sp_d;
float *dg_V_sp_d;
float *dg_Lift_sp_d;
float *dg_Emat_sp_d;
float *dg_Interp_sp_d;

void DGConstants3D::transfer_kernel_ptrs() {
  // Allocate device memory
  cutilSafeCall(cudaMalloc(&dg_r_d, N_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_s_d, N_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_t_d, N_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Dr_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Ds_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Dt_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Drw_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Dsw_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Dtw_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Mass_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_InvMass_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_InvV_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_V_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Lift_d, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_MM_F0_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_MM_F1_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_MM_F2_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_MM_F3_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Emat_d, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Interp_d, N_max * N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_cub3d_Interp_d, DG_NP * DG_CUB_3D_NP * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_cub3d_Proj_d, DG_NP * DG_CUB_3D_NP * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_cub3d_PDr_d, DG_NP * DG_CUB_3D_NP * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_cub3d_PDs_d, DG_NP * DG_CUB_3D_NP * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_cub3d_PDt_d, DG_NP * DG_CUB_3D_NP * sizeof(DG_FP)));

  // Transfer matrices to device
  cutilSafeCall(cudaMemcpy(dg_r_d, r_ptr, N_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_s_d, s_ptr, N_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_t_d, t_ptr, N_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Dr_d, Dr_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Ds_d, Ds_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Dt_d, Dt_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Drw_d, Drw_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Dsw_d, Dsw_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Dtw_d, Dtw_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Mass_d, mass_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_InvMass_d, invMass_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_InvV_d, invV_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_V_d, v_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Lift_d, lift_ptr, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_MM_F0_d, mmF0_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_MM_F1_d, mmF1_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_MM_F2_d, mmF2_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_MM_F3_d, mmF3_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Emat_d, eMat_ptr, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Interp_d, order_interp_ptr, N_max * N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cub3d_Interp_d, cubInterp_ptr, DG_NP * DG_CUB_3D_NP * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cub3d_Proj_d, cubProj_ptr, DG_NP * DG_CUB_3D_NP * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cub3d_PDr_d, cubPDr_ptr, DG_NP * DG_CUB_3D_NP * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cub3d_PDs_d, cubPDs_ptr, DG_NP * DG_CUB_3D_NP * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cub3d_PDt_d, cubPDt_ptr, DG_NP * DG_CUB_3D_NP * sizeof(DG_FP), cudaMemcpyHostToDevice));

  // Set up pointers that are accessible from the device
  cutilSafeCall(cudaMemcpyToSymbol(dg_r_kernel, &dg_r_d, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_s_kernel, &dg_s_d, sizeof(dg_s_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_t_kernel, &dg_t_d, sizeof(dg_t_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Dr_kernel, &dg_Dr_d, sizeof(dg_Dr_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Ds_kernel, &dg_Ds_d, sizeof(dg_Ds_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Dt_kernel, &dg_Dt_d, sizeof(dg_Dt_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Drw_kernel, &dg_Drw_d, sizeof(dg_Drw_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Dsw_kernel, &dg_Dsw_d, sizeof(dg_Dsw_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Dtw_kernel, &dg_Dtw_d, sizeof(dg_Dtw_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Mass_kernel, &dg_Mass_d, sizeof(dg_Mass_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_InvMass_kernel, &dg_InvMass_d, sizeof(dg_InvMass_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_InvV_kernel, &dg_InvV_d, sizeof(dg_InvV_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_V_kernel, &dg_V_d, sizeof(dg_V_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Lift_kernel, &dg_Lift_d, sizeof(dg_Lift_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_MM_F0_kernel, &dg_MM_F0_d, sizeof(dg_MM_F0_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_MM_F1_kernel, &dg_MM_F1_d, sizeof(dg_MM_F1_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_MM_F2_kernel, &dg_MM_F2_d, sizeof(dg_MM_F2_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_MM_F3_kernel, &dg_MM_F3_d, sizeof(dg_MM_F3_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Emat_kernel, &dg_Emat_d, sizeof(dg_Emat_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Interp_kernel, &dg_Interp_d, sizeof(dg_Interp_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_cub3d_Interp_kernel, &dg_cub3d_Interp_d, sizeof(dg_cub3d_Interp_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_cub3d_Proj_kernel, &dg_cub3d_Proj_d, sizeof(dg_cub3d_Proj_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_cub3d_PDr_kernel, &dg_cub3d_PDr_d, sizeof(dg_cub3d_Interp_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_cub3d_PDs_kernel, &dg_cub3d_PDs_d, sizeof(dg_cub3d_PDs_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_cub3d_PDt_kernel, &dg_cub3d_PDt_d, sizeof(dg_cub3d_PDt_d)));

  cutilSafeCall(cudaMalloc(&dg_Dr_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_Ds_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_Dt_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_Drw_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_Dsw_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_Dtw_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_Mass_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_InvMass_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_InvV_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_V_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_Lift_sp_d, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_Emat_sp_d, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_Interp_sp_d, N_max * N_max * Np_max * Np_max * sizeof(float)));

  cutilSafeCall(cudaMemcpy(dg_Dr_sp_d, Dr_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Ds_sp_d, Ds_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Dt_sp_d, Dt_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Drw_sp_d, Drw_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Dsw_sp_d, Dsw_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Dtw_sp_d, Dtw_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Mass_sp_d, mass_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_InvMass_sp_d, invMass_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_InvV_sp_d, invV_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_V_sp_d, v_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Lift_sp_d, lift_ptr_sp, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Emat_sp_d, eMat_ptr_sp, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Interp_sp_d, order_interp_ptr_sp, N_max * N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
}

void DGConstants3D::clean_up_kernel_ptrs() {
  cudaFree(dg_r_d);
  cudaFree(dg_s_d);
  cudaFree(dg_t_d);
  cudaFree(dg_Dr_d);
  cudaFree(dg_Ds_d);
  cudaFree(dg_Dt_d);
  cudaFree(dg_Drw_d);
  cudaFree(dg_Dsw_d);
  cudaFree(dg_Dtw_d);
  cudaFree(dg_Mass_d);
  cudaFree(dg_InvMass_d);
  cudaFree(dg_InvV_d);
  cudaFree(dg_V_d);
  cudaFree(dg_Lift_d);
  cudaFree(dg_MM_F0_d);
  cudaFree(dg_MM_F1_d);
  cudaFree(dg_MM_F2_d);
  cudaFree(dg_MM_F3_d);
  cudaFree(dg_Emat_d);
  cudaFree(dg_Interp_d);

  cudaFree(dg_Dr_sp_d);
  cudaFree(dg_Ds_sp_d);
  cudaFree(dg_Dt_sp_d);
  cudaFree(dg_Drw_sp_d);
  cudaFree(dg_Dsw_sp_d);
  cudaFree(dg_Dtw_sp_d);
  cudaFree(dg_Mass_sp_d);
  cudaFree(dg_InvMass_sp_d);
  cudaFree(dg_InvV_sp_d);
  cudaFree(dg_V_sp_d);
  cudaFree(dg_Lift_sp_d);
  cudaFree(dg_Emat_sp_d);
  cudaFree(dg_Interp_sp_d);
}

DG_FP* DGConstants3D::get_mat_ptr_kernel(Constant_Matrix matrix) {
  switch(matrix) {
    case R:
      return dg_r_d;
    case S:
      return dg_s_d;
    case T:
      return dg_t_d;
    case DR:
      return dg_Dr_d;
    case DS:
      return dg_Ds_d;
    case DT:
      return dg_Dt_d;
    case DRW:
      return dg_Drw_d;
    case DSW:
      return dg_Dsw_d;
    case DTW:
      return dg_Dtw_d;
    case MASS:
      return dg_Mass_d;
    case INV_MASS:
      return dg_InvMass_d;
    case INV_V:
      return dg_InvV_d;
    case V:
      return dg_V_d;
    case LIFT:
      return dg_Lift_d;
    case MM_F0:
      return dg_MM_F0_d;
    case MM_F1:
      return dg_MM_F1_d;
    case MM_F2:
      return dg_MM_F2_d;
    case MM_F3:
      return dg_MM_F3_d;
    case EMAT:
      return dg_Emat_d;
    case INTERP_MATRIX_ARRAY:
      return dg_Interp_d;
    case CUB3D_INTERP:
      return dg_cub3d_Interp_d;
    case CUB3D_PROJ:
      return dg_cub3d_Proj_d;
    case CUB3D_PDR:
      return dg_cub3d_PDr_d;
    case CUB3D_PDS:
      return dg_cub3d_PDs_d;
    case CUB3D_PDT:
      return dg_cub3d_PDt_d;
    default:
      throw std::runtime_error("This constant matrix is not supported by DGConstants3D\n");
      return nullptr;
  }
}

float* DGConstants3D::get_mat_ptr_kernel_sp(Constant_Matrix matrix) {
  switch(matrix) {
    case DR:
      return dg_Dr_sp_d;
    case DS:
      return dg_Ds_sp_d;
    case DT:
      return dg_Dt_sp_d;
    case DRW:
      return dg_Drw_sp_d;
    case DSW:
      return dg_Dsw_sp_d;
    case DTW:
      return dg_Dtw_sp_d;
    case MASS:
      return dg_Mass_sp_d;
    case INV_MASS:
      return dg_InvMass_sp_d;
    case INV_V:
      return dg_InvV_sp_d;
    case V:
      return dg_V_sp_d;
    case LIFT:
      return dg_Lift_sp_d;
    case EMAT:
      return dg_Emat_sp_d;
    case INTERP_MATRIX_ARRAY:
      return dg_Interp_sp_d;
    default:
      throw std::runtime_error("This sp constant matrix is not supported by DGConstants3D\n");
      return nullptr;
  }
}
