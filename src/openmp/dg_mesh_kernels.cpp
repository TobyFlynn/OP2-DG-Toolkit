//
// auto-generated by op2.py
//

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "dg_compiler_defs.h"

// global constants
extern int FMASK[DG_NPF * 3];
extern double cubW_g[DG_CUB_NP];
extern double cubV_g[DG_CUB_NP * DG_NP];
extern double cubVDr_g[DG_CUB_NP * DG_NP];
extern double cubVDs_g[DG_CUB_NP * DG_NP];
extern double gF0Dr_g[DG_GF_NP * DG_NP];
extern double gF0Ds_g[DG_GF_NP * DG_NP];
extern double gF1Dr_g[DG_GF_NP * DG_NP];
extern double gF1Ds_g[DG_GF_NP * DG_NP];
extern double gF2Dr_g[DG_GF_NP * DG_NP];
extern double gF2Ds_g[DG_GF_NP * DG_NP];
extern double gaussW_g[DG_GF_NP];
extern double gFInterp0_g[DG_GF_NP * DG_NP];
extern double gFInterp1_g[DG_GF_NP * DG_NP];
extern double gFInterp2_g[DG_GF_NP * DG_NP];
extern double gF0DrR_g[DG_GF_NP * DG_NP];
extern double gF0DsR_g[DG_GF_NP * DG_NP];
extern double gF1DrR_g[DG_GF_NP * DG_NP];
extern double gF1DsR_g[DG_GF_NP * DG_NP];
extern double gF2DrR_g[DG_GF_NP * DG_NP];
extern double gF2DsR_g[DG_GF_NP * DG_NP];
extern double gFInterp0R_g[DG_GF_NP * DG_NP];
extern double gFInterp1R_g[DG_GF_NP * DG_NP];
extern double gFInterp2R_g[DG_GF_NP * DG_NP];

// header
#include "op_lib_cpp.h"

// user kernel files
#include "init_cubature_kernel.cpp"
#include "init_gauss_kernel.cpp"
#include "init_nodes_kernel.cpp"
#include "init_grid_kernel.cpp"
#include "init_edges_kernel.cpp"
#include "div_kernel.cpp"
#include "curl_kernel.cpp"
#include "grad_kernel.cpp"
#include "cub_grad_kernel.cpp"
#include "cub_div_kernel.cpp"
#include "cub_grad_weak_kernel.cpp"
#include "cub_div_weak_kernel.cpp"
#include "inv_J_kernel.cpp"
