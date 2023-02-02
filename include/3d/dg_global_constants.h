#ifndef __DG_3D_GLOBAL_CONSTANTS_H
#define __DG_3D_GLOBAL_CONSTANTS_H

#include "dg_compiler_defs.h"

extern int FMASK[DG_ORDER * DG_NUM_FACES * DG_NPF];
extern int DG_CONSTANTS[DG_ORDER * DG_NUM_CONSTANTS];

// TODO not require this
extern double cubW_g[1];
extern double gaussW_g[1];

#endif