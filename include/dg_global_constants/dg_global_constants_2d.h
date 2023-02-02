#ifndef __DG_GLOBAL_CONSTANTS_2D_H
#define __DG_GLOBAL_CONSTANTS_2D_H

#include "../dg_compiler_defs.h"

extern int DG_CONSTANTS[DG_ORDER * DG_NUM_CONSTANTS];
extern int FMASK[DG_ORDER * DG_NUM_FACES * DG_NPF];
extern double cubW_g[DG_ORDER * DG_CUB_NP];
extern double gaussW_g[DG_ORDER * DG_GF_NP];

#endif