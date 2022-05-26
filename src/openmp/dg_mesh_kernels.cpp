//
// auto-generated by op2.py
//

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "dg_compiler_defs.h"

// global constants
extern int DG_CONSTANTS[DG_ORDER * 5];
extern int FMASK[DG_ORDER * DG_NPF * 3];
extern double cubW_g[DG_ORDER * DG_CUB_NP];
extern double gaussW_g[DG_ORDER * DG_GF_NP];

// header
#include "op_lib_cpp.h"

// user kernel files
#include "init_cubature_kernel.cpp"
#include "cub_mm_init_kernel.cpp"
#include "init_gauss_kernel.cpp"
#include "init_order_kernel.cpp"
#include "init_nodes_kernel.cpp"
#include "init_grid_kernel.cpp"
#include "init_edges_kernel.cpp"
#include "interp_dat_to_new_order_kernel.cpp"
#include "copy_new_orders_kernel.cpp"
#include "interp_dat_to_new_order_int_kernel.cpp"
#include "copy_new_orders_int_kernel.cpp"
#include "interp_dat_to_max_order_kernel.cpp"
#include "gemv_inv_mass_gauss_interpT_kernel.cpp"
#include "gemv_gauss_interpT_kernel.cpp"
#include "gemv_gauss_interp_kernel.cpp"
#include "gemv_cub_np_npT_kernel.cpp"
#include "gemv_cub_np_np_kernel.cpp"
#include "gemv_np_npT_kernel.cpp"
#include "gemv_np_np_kernel.cpp"
#include "gemv_liftT_kernel.cpp"
#include "gemv_lift_kernel.cpp"
#include "div_kernel.cpp"
#include "curl_kernel.cpp"
#include "grad_kernel.cpp"
#include "cub_grad_kernel.cpp"
#include "cub_div_kernel.cpp"
#include "cub_grad_weak_kernel.cpp"
#include "cub_div_weak_kernel.cpp"
#include "inv_J_kernel.cpp"
