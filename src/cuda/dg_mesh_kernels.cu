//
// auto-generated by op2.py
//

//global constants
#ifndef MAX_CONST_SIZE
#define MAX_CONST_SIZE 128
#endif

#include "dg_compiler_defs.h"

__constant__ int FMASK_cuda[DG_NP];
__constant__ double cubW_g_cuda[DG_CUB_NP];
__constant__ double cubV_g_cuda[DG_CUB_NP * DG_NP];
__constant__ double cubVDr_g_cuda[DG_CUB_NP * DG_NP];
__constant__ double cubVDs_g_cuda[DG_CUB_NP * DG_NP];
__constant__ double gF0Dr_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gF0Ds_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gF1Dr_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gF1Ds_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gF2Dr_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gF2Ds_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gaussW_g_cuda[DG_GF_NP];
__constant__ double gFInterp0_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gFInterp1_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gFInterp2_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gF0DrR_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gF0DsR_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gF1DrR_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gF1DsR_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gF2DrR_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gF2DsR_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gFInterp0R_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gFInterp1R_g_cuda[DG_GF_NP * DG_NP];
__constant__ double gFInterp2R_g_cuda[DG_GF_NP * DG_NP];

//header
#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

#include "dg_global_constants.h"

void set_cuda_const() {
  cutilSafeCall(cudaMemcpyToSymbol(FMASK_cuda, FMASK, DG_NP * sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(cubW_g_cuda, cubW_g, DG_CUB_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(cubV_g_cuda, cubV_g, DG_CUB_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(cubVDr_g_cuda, cubVDr_g, DG_CUB_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(cubVDs_g_cuda, cubVDs_g, DG_CUB_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gF0Dr_g_cuda, gF0Dr_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gF0Ds_g_cuda, gF0Ds_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gF1Dr_g_cuda, gF1Dr_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gF1Ds_g_cuda, gF1Ds_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gF2Dr_g_cuda, gF2Dr_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gF2Ds_g_cuda, gF2Ds_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gaussW_g_cuda, gaussW_g, DG_GF_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gFInterp0_g_cuda, gFInterp0_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gFInterp1_g_cuda, gFInterp1_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gFInterp2_g_cuda, gFInterp2_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gF0DrR_g_cuda, gF0DrR_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gF0DsR_g_cuda, gF0DsR_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gF1DrR_g_cuda, gF1DrR_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gF1DsR_g_cuda, gF1DsR_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gF2DrR_g_cuda, gF2DrR_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gF2DsR_g_cuda, gF2DsR_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gFInterp0R_g_cuda, gFInterp0R_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gFInterp1R_g_cuda, gFInterp1R_g, DG_GF_NP * DG_NP * sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(gFInterp2R_g_cuda, gFInterp2R_g, DG_GF_NP * DG_NP * sizeof(double)));
}

//user kernel files
#include "init_cubature_kernel.cu"
#include "init_gauss_kernel.cu"
#include "init_nodes_kernel.cu"
#include "init_grid_kernel.cu"
#include "init_edges_kernel.cu"
#include "div_kernel.cu"
#include "curl_kernel.cu"
#include "grad_kernel.cu"
#include "cub_grad_kernel.cu"
#include "cub_div_kernel.cu"
#include "cub_grad_weak_kernel.cu"
#include "cub_div_weak_kernel.cu"
#include "inv_J_kernel.cu"
