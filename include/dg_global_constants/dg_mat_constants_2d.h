#ifndef __DG_2D_MAT_CONSTANTS_H
#define __DG_2D_MAT_CONSTANTS_H

#include "dg_compiler_defs.h"

#ifdef OP2_DG_CUDA
extern __constant__ DG_FP *dg_r_kernel;
extern __constant__ DG_FP *dg_s_kernel;
extern __constant__ DG_FP *dg_Dr_kernel;
extern __constant__ DG_FP *dg_Ds_kernel;
extern __constant__ DG_FP *dg_Drw_kernel;
extern __constant__ DG_FP *dg_Dsw_kernel;
extern __constant__ DG_FP *dg_Mass_kernel;
extern __constant__ DG_FP *dg_InvMass_kernel;
extern __constant__ DG_FP *dg_InvV_kernel;
extern __constant__ DG_FP *dg_Lift_kernel;
extern __constant__ DG_FP *dg_Interp_kernel;
#else
extern DG_FP *dg_r_kernel;
extern DG_FP *dg_s_kernel;
extern DG_FP *dg_Dr_kernel;
extern DG_FP *dg_Ds_kernel;
extern DG_FP *dg_Drw_kernel;
extern DG_FP *dg_Dsw_kernel;
extern DG_FP *dg_Mass_kernel;
extern DG_FP *dg_InvMass_kernel;
extern DG_FP *dg_InvV_kernel;
extern DG_FP *dg_Lift_kernel;
extern DG_FP *dg_Interp_kernel;
#endif

#endif
