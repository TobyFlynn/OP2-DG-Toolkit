#include "dg_constants/dg_constants_2d.h"

#include "dg_abort.h"
#include "op_cuda_rt_support.h"

__constant__ DG_FP *dg_r_kernel;
__constant__ DG_FP *dg_s_kernel;
__constant__ DG_FP *dg_Dr_kernel;
__constant__ DG_FP *dg_Ds_kernel;
__constant__ DG_FP *dg_Drw_kernel;
__constant__ DG_FP *dg_Dsw_kernel;
__constant__ DG_FP *dg_Mass_kernel;
__constant__ DG_FP *dg_InvMass_kernel;
__constant__ DG_FP *dg_InvV_kernel;
__constant__ DG_FP *dg_V_kernel;
__constant__ DG_FP *dg_Lift_kernel;
__constant__ DG_FP *dg_Interp_kernel;
__constant__ DG_FP *dg_MM_F0_kernel;
__constant__ DG_FP *dg_MM_F1_kernel;
__constant__ DG_FP *dg_MM_F2_kernel;
__constant__ DG_FP *dg_Emat_kernel;
__constant__ DG_FP *dg_cubSurf2d_Interp_kernel;

DG_FP *dg_r_d;
DG_FP *dg_s_d;
DG_FP *dg_Dr_d;
DG_FP *dg_Ds_d;
DG_FP *dg_Drw_d;
DG_FP *dg_Dsw_d;
DG_FP *dg_Mass_d;
DG_FP *dg_InvMass_d;
DG_FP *dg_InvV_d;
DG_FP *dg_V_d;
DG_FP *dg_Lift_d;
DG_FP *dg_Interp_d;
DG_FP *dg_MM_F0_d;
DG_FP *dg_MM_F1_d;
DG_FP *dg_MM_F2_d;
DG_FP *dg_Emat_d;
DG_FP *dg_cubInterp_d;
DG_FP *dg_cubProj_d;
DG_FP *dg_cubPDrT_d;
DG_FP *dg_cubPDsT_d;
DG_FP *dg_cubInterpSurf_d;
DG_FP *dg_cubLiftSurf_d;

float *dg_Dr_sp_d;
float *dg_Ds_sp_d;
float *dg_Drw_sp_d;
float *dg_Dsw_sp_d;
float *dg_Mass_sp_d;
float *dg_InvMass_sp_d;
float *dg_InvV_sp_d;
float *dg_V_sp_d;
float *dg_Lift_sp_d;
float *dg_Emat_sp_d;
float *dg_Interp_sp_d;
float *dg_cubInterp_sp_d;
float *dg_cubProj_sp_d;
float *dg_cubPDrT_sp_d;
float *dg_cubPDsT_sp_d;
float *dg_cubInterpSurf_sp_d;
float *dg_cubLiftSurf_sp_d;

void DGConstants2D::transfer_kernel_ptrs() {
  // Allocate device memory
  cutilSafeCall(cudaMalloc(&dg_r_d, N_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_s_d, N_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Dr_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Ds_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Drw_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Dsw_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Mass_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_InvMass_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_InvV_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_V_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Lift_d, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Interp_d, N_max * N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_MM_F0_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_MM_F1_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_MM_F2_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Emat_d, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_cubInterp_d, DG_NP * DG_CUB_2D_NP * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_cubProj_d, DG_NP * DG_CUB_2D_NP * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_cubPDrT_d, DG_NP * DG_CUB_2D_NP * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_cubPDsT_d, DG_NP * DG_CUB_2D_NP * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_cubInterpSurf_d, DG_NUM_FACES * DG_NPF * DG_NUM_FACES * DG_CUB_SURF_2D_NP * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_cubLiftSurf_d, DG_NP * DG_NUM_FACES * DG_CUB_SURF_2D_NP * sizeof(DG_FP)));

  // Transfer matrices to device
  cutilSafeCall(cudaMemcpy(dg_r_d, r_ptr, N_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_s_d, s_ptr, N_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Dr_d, Dr_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Ds_d, Ds_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Drw_d, Drw_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Dsw_d, Dsw_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Mass_d, mass_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_InvMass_d, invMass_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_InvV_d, invV_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_V_d, v_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Lift_d, lift_ptr, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Interp_d, order_interp_ptr, N_max * N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_MM_F0_d, mmF0_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_MM_F1_d, mmF1_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_MM_F2_d, mmF2_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Emat_d, eMat_ptr, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cubInterp_d, cubInterp_ptr, DG_NP * DG_CUB_2D_NP * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cubProj_d, cubProj_ptr, DG_NP * DG_CUB_2D_NP * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cubPDrT_d, cubPDrT_ptr, DG_NP * DG_CUB_2D_NP * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cubPDsT_d, cubPDsT_ptr, DG_NP * DG_CUB_2D_NP * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cubInterpSurf_d, cubInterpSurf_ptr, DG_NUM_FACES * DG_NPF * DG_NUM_FACES * DG_CUB_SURF_2D_NP * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cubLiftSurf_d, cubLiftSurf_ptr, DG_NP * DG_NUM_FACES * DG_CUB_SURF_2D_NP * sizeof(DG_FP), cudaMemcpyHostToDevice));

  // Set up pointers that are accessible from the device
  cutilSafeCall(cudaMemcpyToSymbol(dg_r_kernel, &dg_r_d, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_s_kernel, &dg_s_d, sizeof(dg_s_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Dr_kernel, &dg_Dr_d, sizeof(dg_Dr_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Ds_kernel, &dg_Ds_d, sizeof(dg_Ds_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Drw_kernel, &dg_Drw_d, sizeof(dg_Drw_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Dsw_kernel, &dg_Dsw_d, sizeof(dg_Dsw_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Mass_kernel, &dg_Mass_d, sizeof(dg_Mass_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_InvMass_kernel, &dg_InvMass_d, sizeof(dg_InvMass_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_InvV_kernel, &dg_InvV_d, sizeof(dg_InvV_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_V_kernel, &dg_V_d, sizeof(dg_V_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Lift_kernel, &dg_Lift_d, sizeof(dg_Lift_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Interp_kernel, &dg_Interp_d, sizeof(dg_Interp_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_MM_F0_kernel, &dg_MM_F0_d, sizeof(dg_MM_F0_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_MM_F1_kernel, &dg_MM_F1_d, sizeof(dg_MM_F1_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_MM_F2_kernel, &dg_MM_F2_d, sizeof(dg_MM_F2_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Emat_kernel, &dg_Emat_d, sizeof(dg_Emat_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_cubSurf2d_Interp_kernel, &dg_cubInterpSurf_d, sizeof(dg_cubInterpSurf_d)));

  cutilSafeCall(cudaMalloc(&dg_Dr_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_Ds_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_Drw_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_Dsw_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_Mass_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_InvMass_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_InvV_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_V_sp_d, N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_Lift_sp_d, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_Emat_sp_d, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_Interp_sp_d, N_max * N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_cubInterp_sp_d, DG_NP * DG_CUB_2D_NP * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_cubProj_sp_d, DG_NP * DG_CUB_2D_NP * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_cubPDrT_sp_d, DG_NP * DG_CUB_2D_NP * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_cubPDsT_sp_d, DG_NP * DG_CUB_2D_NP * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_cubInterpSurf_sp_d, DG_NUM_FACES * DG_NPF * DG_NUM_FACES * DG_CUB_SURF_2D_NP * sizeof(float)));
  cutilSafeCall(cudaMalloc(&dg_cubLiftSurf_sp_d, DG_NP * DG_NUM_FACES * DG_CUB_SURF_2D_NP * sizeof(float)));

  cutilSafeCall(cudaMemcpy(dg_Dr_sp_d, Dr_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Ds_sp_d, Ds_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Drw_sp_d, Drw_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Dsw_sp_d, Dsw_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Mass_sp_d, mass_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_InvMass_sp_d, invMass_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_InvV_sp_d, invV_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_V_sp_d, v_ptr_sp, N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Lift_sp_d, lift_ptr_sp, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Emat_sp_d, eMat_ptr_sp, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Interp_sp_d, order_interp_ptr_sp, N_max * N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cubInterp_sp_d, cubInterp_ptr_sp, DG_NP * DG_CUB_2D_NP * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cubProj_sp_d, cubProj_ptr_sp, DG_NP * DG_CUB_2D_NP * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cubPDrT_sp_d, cubPDrT_ptr_sp, DG_NP * DG_CUB_2D_NP * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cubPDsT_sp_d, cubPDsT_ptr_sp, DG_NP * DG_CUB_2D_NP * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cubInterpSurf_sp_d, cubInterpSurf_ptr_sp, DG_NUM_FACES * DG_NPF * DG_NUM_FACES * DG_CUB_SURF_2D_NP * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_cubLiftSurf_sp_d, cubLiftSurf_ptr_sp, DG_NP * DG_NUM_FACES * DG_CUB_SURF_2D_NP * sizeof(float), cudaMemcpyHostToDevice));
}

void DGConstants2D::clean_up_kernel_ptrs() {
  cudaFree(dg_r_d);
  cudaFree(dg_s_d);
  cudaFree(dg_Dr_d);
  cudaFree(dg_Ds_d);
  cudaFree(dg_Drw_d);
  cudaFree(dg_Dsw_d);
  cudaFree(dg_Mass_d);
  cudaFree(dg_InvMass_d);
  cudaFree(dg_InvV_d);
  cudaFree(dg_V_d);
  cudaFree(dg_Lift_d);
  cudaFree(dg_Interp_d);
  cudaFree(dg_MM_F0_d);
  cudaFree(dg_MM_F1_d);
  cudaFree(dg_MM_F2_d);
  cudaFree(dg_Emat_d);
  cudaFree(dg_cubInterp_d);
  cudaFree(dg_cubProj_d);
  cudaFree(dg_cubPDrT_d);
  cudaFree(dg_cubPDsT_d);
  cudaFree(dg_cubInterpSurf_d);
  cudaFree(dg_cubLiftSurf_d);

  cudaFree(dg_Dr_sp_d);
  cudaFree(dg_Ds_sp_d);
  cudaFree(dg_Drw_sp_d);
  cudaFree(dg_Dsw_sp_d);
  cudaFree(dg_Mass_sp_d);
  cudaFree(dg_InvMass_sp_d);
  cudaFree(dg_InvV_sp_d);
  cudaFree(dg_V_sp_d);
  cudaFree(dg_Lift_sp_d);
  cudaFree(dg_Emat_sp_d);
  cudaFree(dg_Interp_sp_d);
  cudaFree(dg_cubInterp_sp_d);
  cudaFree(dg_cubProj_sp_d);
  cudaFree(dg_cubPDrT_sp_d);
  cudaFree(dg_cubPDsT_sp_d);
  cudaFree(dg_cubInterpSurf_sp_d);
  cudaFree(dg_cubLiftSurf_sp_d);
}

DG_FP* DGConstants2D::get_mat_ptr_device(Constant_Matrix matrix) {
  switch(matrix) {
    case R:
      return dg_r_d;
    case S:
      return dg_s_d;
    case DR:
      return dg_Dr_d;
    case DS:
      return dg_Ds_d;
    case DRW:
      return dg_Drw_d;
    case DSW:
      return dg_Dsw_d;
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
    case INTERP_MATRIX_ARRAY:
      return dg_Interp_d;
    case MM_F0:
      return dg_MM_F0_d;
    case MM_F1:
      return dg_MM_F1_d;
    case MM_F2:
      return dg_MM_F2_d;
    case EMAT:
      return dg_Emat_d;
    case CUB2D_INTERP:
      return dg_cubInterp_d;
    case CUB2D_PROJ:
      return dg_cubProj_d;
    case CUB2D_PDR:
      return dg_cubPDrT_d;
    case CUB2D_PDS:
      return dg_cubPDsT_d;
    case CUBSURF2D_INTERP:
      return dg_cubInterpSurf_d;
    case CUBSURF2D_LIFT:
      return dg_cubLiftSurf_d;
    default:
      dg_abort("This constant matrix is not supported by DGConstants2D\n");
      return nullptr;
  }
}

float* DGConstants2D::get_mat_ptr_device_sp(Constant_Matrix matrix) {
  switch(matrix) {
    case DR:
      return dg_Dr_sp_d;
    case DS:
      return dg_Ds_sp_d;
    case DRW:
      return dg_Drw_sp_d;
    case DSW:
      return dg_Dsw_sp_d;
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
    case CUB2D_INTERP:
      return dg_cubInterp_sp_d;
    case CUB2D_PROJ:
      return dg_cubProj_sp_d;
    case CUB2D_PDR:
      return dg_cubPDrT_sp_d;
    case CUB2D_PDS:
      return dg_cubPDsT_sp_d;
    case CUBSURF2D_INTERP:
      return dg_cubInterpSurf_sp_d;
    case CUBSURF2D_LIFT:
      return dg_cubLiftSurf_sp_d;
    default:
      dg_abort("This sp constant matrix is not supported by DGConstants2D\n");
      return nullptr;
  }
}
