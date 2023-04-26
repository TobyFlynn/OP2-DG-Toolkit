#ifndef __DG_3D_GLOBAL_CONSTANTS_H
#define __DG_3D_GLOBAL_CONSTANTS_H

#include "dg_compiler_defs.h"

extern int FMASK[DG_ORDER * DG_NUM_FACES * DG_NPF];
extern int DG_CONSTANTS[DG_ORDER * DG_NUM_CONSTANTS];

extern int FMASK_TK[DG_ORDER * DG_NUM_FACES * DG_NPF];
extern int DG_CONSTANTS_TK[DG_ORDER * DG_NUM_CONSTANTS];

// TODO not require this
extern DG_FP cubW_g[1];
extern DG_FP gaussW_g[1];

#endif
