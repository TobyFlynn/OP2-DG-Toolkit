//
// auto-generated by op2.py
//

// global constants
extern double gam;
extern double mu;
extern double nu0;
extern double nu1;
extern double rho0;
extern double rho1;
extern double ren;
extern double bc_mach;
extern double bc_alpha;
extern double bc_p;
extern double bc_u;
extern double bc_v;
extern int FMASK[15];
extern double ic_u;
extern double ic_v;
extern double cubW_g[46];
extern double cubV_g[690];
extern double cubVDr_g[690];
extern double cubVDs_g[690];
extern double gF0Dr_g[105];
extern double gF0Ds_g[105];
extern double gF1Dr_g[105];
extern double gF1Ds_g[105];
extern double gF2Dr_g[105];
extern double gF2Ds_g[105];
extern double gaussW_g[7];
extern double gFInterp0_g[105];
extern double gFInterp1_g[105];
extern double gFInterp2_g[105];
extern double gF0DrR_g[105];
extern double gF0DsR_g[105];
extern double gF1DrR_g[105];
extern double gF1DsR_g[105];
extern double gF2DrR_g[105];
extern double gF2DsR_g[105];
extern double gFInterp0R_g[105];
extern double gFInterp1R_g[105];
extern double gFInterp2R_g[105];
extern double lift_drag_vec[5];

// header
#include "op_lib_c.h"

void op_decl_const_char(int dim, char const *type,
int size, char *dat, char const *name){}
// user kernel files
#include "init_cubature_acckernel.c"
#include "init_gauss_acckernel.c"
#include "init_nodes_acckernel.c"
#include "init_grid_acckernel.c"
#include "init_edges_acckernel.c"
#include "div_acckernel.c"
#include "curl_acckernel.c"
#include "grad_acckernel.c"
#include "cub_grad_acckernel.c"
#include "cub_div_acckernel.c"
#include "cub_grad_weak_acckernel.c"
#include "cub_div_weak_acckernel.c"
#include "inv_J_acckernel.c"
