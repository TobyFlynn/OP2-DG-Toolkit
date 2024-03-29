#include "dg_constants/dg_constants_2d.h"

#include "dg_abort.h"
#include "op_hip_rt_support.h"

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

void DGConstants2D::transfer_kernel_ptrs() {
  // Allocate device memory
  cutilSafeCall(hipMalloc(&dg_r_d, N_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(hipMalloc(&dg_s_d, N_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(hipMalloc(&dg_Dr_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(hipMalloc(&dg_Ds_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(hipMalloc(&dg_Drw_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(hipMalloc(&dg_Dsw_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(hipMalloc(&dg_Mass_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(hipMalloc(&dg_InvMass_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(hipMalloc(&dg_InvV_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(hipMalloc(&dg_V_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(hipMalloc(&dg_Lift_d, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(hipMalloc(&dg_Interp_d, N_max * N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(hipMalloc(&dg_MM_F0_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(hipMalloc(&dg_MM_F1_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(hipMalloc(&dg_MM_F2_d, N_max * Np_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(hipMalloc(&dg_Emat_d, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(DG_FP)));

  // Transfer matrices to device
  cutilSafeCall(hipMemcpy(dg_r_d, r_ptr, N_max * Np_max * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_s_d, s_ptr, N_max * Np_max * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_Dr_d, Dr_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_Ds_d, Ds_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_Drw_d, Drw_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_Dsw_d, Dsw_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_Mass_d, mass_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_InvMass_d, invMass_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_InvV_d, invV_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_V_d, v_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_Lift_d, lift_ptr, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_Interp_d, order_interp_ptr, N_max * N_max * Np_max * Np_max * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_MM_F0_d, mmF0_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_MM_F1_d, mmF1_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_MM_F2_d, mmF2_ptr, N_max * Np_max * Np_max * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_Emat_d, eMat_ptr, N_max * DG_NUM_FACES * Nfp_max * Np_max * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_cubInterp_d, cubInterp_ptr, DG_NP * DG_CUB_2D_NP * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_cubProj_d, cubProj_ptr, DG_NP * DG_CUB_2D_NP * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_cubPDrT_d, cubPDrT_ptr, DG_NP * DG_CUB_2D_NP * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_cubPDsT_d, cubPDsT_ptr, DG_NP * DG_CUB_2D_NP * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_cubInterpSurf_d, cubInterpSurf_ptr, DG_NUM_FACES * DG_NPF * DG_NUM_FACES * DG_CUB_SURF_2D_NP * sizeof(DG_FP), hipMemcpyHostToDevice));
  cutilSafeCall(hipMemcpy(dg_cubLiftSurf_d, cubLiftSurf_ptr, DG_NP * DG_NUM_FACES * DG_CUB_SURF_2D_NP * sizeof(DG_FP), hipMemcpyHostToDevice));

  // Set up pointers that are accessible from the device
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_r_kernel), &dg_r_d, sizeof(dg_r_d)));
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_s_kernel), &dg_s_d, sizeof(dg_s_d)));
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_Dr_kernel), &dg_Dr_d, sizeof(dg_Dr_d)));
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_Ds_kernel), &dg_Ds_d, sizeof(dg_Ds_d)));
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_Drw_kernel), &dg_Drw_d, sizeof(dg_Drw_d)));
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_Dsw_kernel), &dg_Dsw_d, sizeof(dg_Dsw_d)));
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_Mass_kernel), &dg_Mass_d, sizeof(dg_Mass_d)));
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_InvMass_kernel), &dg_InvMass_d, sizeof(dg_InvMass_d)));
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_InvV_kernel), &dg_InvV_d, sizeof(dg_InvV_d)));
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_V_kernel), &dg_V_d, sizeof(dg_V_d)));
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_Lift_kernel), &dg_Lift_d, sizeof(dg_Lift_d)));
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_Interp_kernel), &dg_Interp_d, sizeof(dg_Interp_d)));
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_MM_F0_kernel), &dg_MM_F0_d, sizeof(dg_MM_F0_d)));
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_MM_F1_kernel), &dg_MM_F1_d, sizeof(dg_MM_F1_d)));
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_MM_F2_kernel), &dg_MM_F2_d, sizeof(dg_MM_F2_d)));
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_Emat_kernel), &dg_Emat_d, sizeof(dg_Emat_d)));
  cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(dg_cubSurf2d_Interp_kernel), &dg_cubInterpSurf_d, sizeof(dg_cubInterpSurf_d)));
}

void DGConstants2D::clean_up_kernel_ptrs() {
  hipFree(dg_r_d);
  hipFree(dg_s_d);
  hipFree(dg_Dr_d);
  hipFree(dg_Ds_d);
  hipFree(dg_Drw_d);
  hipFree(dg_Dsw_d);
  hipFree(dg_Mass_d);
  hipFree(dg_InvMass_d);
  hipFree(dg_InvV_d);
  hipFree(dg_V_d);
  hipFree(dg_Lift_d);
  hipFree(dg_Interp_d);
  hipFree(dg_MM_F0_kernel);
  hipFree(dg_MM_F1_kernel);
  hipFree(dg_MM_F2_kernel);
  hipFree(dg_Emat_d);
  hipFree(dg_cubInterp_d);
  hipFree(dg_cubProj_d);
  hipFree(dg_cubPDrT_d);
  hipFree(dg_cubPDsT_d);
  hipFree(dg_cubInterpSurf_d);
  hipFree(dg_cubLiftSurf_d);
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
  dg_abort("get_mat_ptr_device_sp is not implemented yet for 2D\n");
  return nullptr;
}
