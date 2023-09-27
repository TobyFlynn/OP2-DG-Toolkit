#ifndef __DG_2D_MAT_CONSTANTS_DEV_PTRS_H
#define __DG_2D_MAT_CONSTANTS_DEV_PTRS_H

#if defined(OP2_DG_CUDA) || defined(OP2_DG_HIP)
#include "dg_compiler_defs.h"

extern DG_FP *dg_r_d;
extern DG_FP *dg_s_d;
extern DG_FP *dg_Dr_d;
extern DG_FP *dg_Ds_d;
extern DG_FP *dg_Drw_d;
extern DG_FP *dg_Dsw_d;
extern DG_FP *dg_Mass_d;
extern DG_FP *dg_InvMass_d;
extern DG_FP *dg_InvV_d;
extern DG_FP *dg_V_d;
extern DG_FP *dg_Lift_d;
extern DG_FP *dg_Interp_d;
extern DG_FP *dg_MM_F0_d;
extern DG_FP *dg_MM_F1_d;
extern DG_FP *dg_MM_F2_d;
extern DG_FP *dg_Emat_d;
#endif

#endif
