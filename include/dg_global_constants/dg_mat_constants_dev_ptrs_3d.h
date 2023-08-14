#ifndef __DG_3D_MAT_CONSTANTS_DEV_PTRS_H
#define __DG_3D_MAT_CONSTANTS_DEV_PTRS_H

#ifdef OP2_DG_CUDA
#include "dg_compiler_defs.h"

extern DG_FP *dg_r_d;
extern DG_FP *dg_s_d;
extern DG_FP *dg_t_d;
extern DG_FP *dg_Dr_d;
extern DG_FP *dg_Ds_d;
extern DG_FP *dg_Dt_d;
extern DG_FP *dg_Drw_d;
extern DG_FP *dg_Dsw_d;
extern DG_FP *dg_Dtw_d;
extern DG_FP *dg_Mass_d;
extern DG_FP *dg_InvMass_d;
extern DG_FP *dg_InvV_d;
extern DG_FP *dg_V_d;
extern DG_FP *dg_Lift_d;
extern DG_FP *dg_MM_F0_d;
extern DG_FP *dg_MM_F1_d;
extern DG_FP *dg_MM_F2_d;
extern DG_FP *dg_MM_F3_d;
extern DG_FP *dg_Emat_d;
extern DG_FP *dg_Interp_d;
extern DG_FP *dg_cub3d_Interp_d;
extern DG_FP *dg_cub3d_Proj_d;
extern DG_FP *dg_cub3d_PDr_d;
extern DG_FP *dg_cub3d_PDs_d;
extern DG_FP *dg_cub3d_PDt_d;
extern DG_FP *dg_cubSurf3d_Interp_d;
extern DG_FP *dg_cubSurf3d_Lift_d;
#endif

#endif
