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
#include "op_lib_cpp.h"

// user kernel files
#include "init_nodes_seqkernel.cpp"
#include "init_grid_seqkernel.cpp"
#include "init_edges_seqkernel.cpp"
