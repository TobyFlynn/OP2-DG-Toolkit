#include "dg_constants/dg_constants_2d.h"

#include "op_hip_rt_support.h"

#include <stdexcept>

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

void DGConstants2D::transfer_kernel_ptrs() {
  throw std::runtime_error("DGConstants2D::transfer_kernel_ptrs not implemented for HIP yet");
  /*
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
  */
}

void DGConstants2D::clean_up_kernel_ptrs() {
  throw std::runtime_error("DGConstants2D::clean_up_kernel_ptrs not implemented for HIP yet");
  /*
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
  cudaFree(dg_MM_F0_kernel);
  cudaFree(dg_MM_F1_kernel);
  cudaFree(dg_MM_F2_kernel);
  cudaFree(dg_Emat_d);
  */
}

DG_FP* DGConstants2D::get_mat_ptr_kernel(Constant_Matrix matrix) {
  throw std::runtime_error("DGConstants2D::get_mat_ptr_kernel not implemented for HIP yet");
  return nullptr;
  /*
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
    default:
      throw std::runtime_error("This constant matrix is not supported by DGConstants2D\n");
      return nullptr;
  }
  */
}

float* DGConstants2D::get_mat_ptr_kernel_sp(Constant_Matrix matrix) {
  throw std::runtime_error("get_mat_ptr_kernel_sp is not implemented yet for 2D\n");
  return nullptr;
}
