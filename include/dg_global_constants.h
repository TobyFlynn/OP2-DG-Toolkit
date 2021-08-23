#ifndef __DG_GLOBAL_CONSTANTS_H
#define __DG_GLOBAL_CONSTANTS_H

#include "dg_compiler_defs.h"

extern int FMASK[3 * DG_NPF];

extern double cubV_g[DG_CUB_NP * DG_NP];
extern double cubW_g[DG_CUB_NP];
extern double cubDr_g[DG_CUB_NP * DG_NP];
extern double cubDs_g[DG_CUB_NP * DG_NP];
extern double cubVDr_g[DG_CUB_NP * DG_NP];
extern double cubVDs_g[DG_CUB_NP * DG_NP];

extern double gaussW_g[DG_GF_NP];
extern double gFInterp0_g[DG_GF_NP * DG_NP];
extern double gFInterp1_g[DG_GF_NP * DG_NP];
extern double gFInterp2_g[DG_GF_NP * DG_NP];
extern double gF0Dr_g[DG_GF_NP * DG_NP];
extern double gF0Ds_g[DG_GF_NP * DG_NP];
extern double gF1Dr_g[DG_GF_NP * DG_NP];
extern double gF1Ds_g[DG_GF_NP * DG_NP];
extern double gF2Dr_g[DG_GF_NP * DG_NP];
extern double gF2Ds_g[DG_GF_NP * DG_NP];
extern double gFInterp0R_g[DG_GF_NP * DG_NP];
extern double gFInterp1R_g[DG_GF_NP * DG_NP];
extern double gFInterp2R_g[DG_GF_NP * DG_NP];
extern double gF0DrR_g[DG_GF_NP * DG_NP];
extern double gF0DsR_g[DG_GF_NP * DG_NP];
extern double gF1DrR_g[DG_GF_NP * DG_NP];
extern double gF1DsR_g[DG_GF_NP * DG_NP];
extern double gF2DrR_g[DG_GF_NP * DG_NP];
extern double gF2DsR_g[DG_GF_NP * DG_NP];
extern double gInterp_g[DG_G_NP* DG_NP];

extern double Dr_g[DG_NP * DG_NP];
extern double Drw_g[DG_NP * DG_NP];
extern double Ds_g[DG_NP * DG_NP];
extern double Dsw_g[DG_NP * DG_NP];
extern double invMass_g[DG_NP * DG_NP];
extern double lift_g[DG_NP * 3 * DG_NPF];
extern double mass_g[DG_NP * DG_NP];
extern double v_g[DG_NP * DG_NP];
extern double invV_g[DG_NP * DG_NP];
extern double r_g[DG_NP];
extern double s_g[DG_NP];
extern double ones_g[DG_NP];

#endif
