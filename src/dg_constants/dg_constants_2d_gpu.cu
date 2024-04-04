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
DG_FP *dg_Interp_d;

float *dg_Interp_sp_d;

void DGConstants2D::transfer_kernel_ptrs() {
  // Allocate device memory
  cutilSafeCall(cudaMalloc(&dg_r_d, N_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_s_d, N_max * Np_max * sizeof(DG_FP)));
  cutilSafeCall(cudaMalloc(&dg_Interp_d, N_max * N_max * Np_max * Np_max * sizeof(DG_FP)));

  // Transfer matrices to device
  cutilSafeCall(cudaMemcpy(dg_r_d, get_mat_ptr(R), N_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_s_d, get_mat_ptr(S), N_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dg_Interp_d, get_mat_ptr(INTERP_MATRIX_ARRAY), N_max * N_max * Np_max * Np_max * sizeof(DG_FP), cudaMemcpyHostToDevice));

  // Set up pointers that are accessible from the device
  cutilSafeCall(cudaMemcpyToSymbol(dg_r_kernel, &dg_r_d, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_s_kernel, &dg_s_d, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Dr_kernel, &dg_mats.at(DR)->mat_ptr_dp_device, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Ds_kernel, &dg_mats.at(DS)->mat_ptr_dp_device, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Drw_kernel, &dg_mats.at(DRW)->mat_ptr_dp_device, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Dsw_kernel, &dg_mats.at(DSW)->mat_ptr_dp_device, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Mass_kernel, &dg_mats.at(MASS)->mat_ptr_dp_device, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_InvMass_kernel, &dg_mats.at(INV_MASS)->mat_ptr_dp_device, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_InvV_kernel, &dg_mats.at(INV_V)->mat_ptr_dp_device, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_V_kernel, &dg_mats.at(V)->mat_ptr_dp_device, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Lift_kernel, &dg_mats.at(LIFT)->mat_ptr_dp_device, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Interp_kernel, &dg_Interp_d, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_MM_F0_kernel, &dg_mats.at(MM_F0)->mat_ptr_dp_device, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_MM_F1_kernel, &dg_mats.at(MM_F1)->mat_ptr_dp_device, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_MM_F2_kernel, &dg_mats.at(MM_F2)->mat_ptr_dp_device, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_Emat_kernel, &dg_mats.at(EMAT)->mat_ptr_dp_device, sizeof(dg_r_d)));
  cutilSafeCall(cudaMemcpyToSymbol(dg_cubSurf2d_Interp_kernel, &dg_mats.at(CUBSURF2D_INTERP)->mat_ptr_dp_device, sizeof(dg_r_d)));

  cutilSafeCall(cudaMalloc(&dg_Interp_sp_d, N_max * N_max * Np_max * Np_max * sizeof(float)));
  cutilSafeCall(cudaMemcpy(dg_Interp_sp_d, order_interp_ptr_sp, N_max * N_max * Np_max * Np_max * sizeof(float), cudaMemcpyHostToDevice));
}

void DGConstants2D::clean_up_kernel_ptrs() {
  cudaFree(dg_r_d);
  cudaFree(dg_s_d);
  cudaFree(dg_Interp_d);

  cudaFree(dg_Interp_sp_d);
}

DG_FP* DGConstants2D::get_mat_ptr_device(Constant_Matrix matrix) {
  switch(matrix) {
    case R:
      return dg_r_d;
    case S:
      return dg_s_d;
    case INTERP_MATRIX_ARRAY:
      return dg_Interp_d;
    default:
      try {
        return dg_mats.at(matrix)->get_mat_ptr_dp_device();
      } catch (std::out_of_range &e) {
        dg_abort("This constant matrix is not supported by DGConstants2D\n");
      }
      return nullptr;
  }
}

float* DGConstants2D::get_mat_ptr_device_sp(Constant_Matrix matrix) {
  switch(matrix) {
    case INTERP_MATRIX_ARRAY:
      return dg_Interp_sp_d;
    default:
      try {
        return dg_mats.at(matrix)->get_mat_ptr_sp_device();
      } catch (std::out_of_range &e) {
        dg_abort("This single precision constant matrix is not supported by DGConstants2D\n");
      }
      return nullptr;
  }
}
