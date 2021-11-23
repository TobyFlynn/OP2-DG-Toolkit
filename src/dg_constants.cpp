#include "dg_constants.h"

#include "dg_global_constants.h"

DGConstants::DGConstants(const int n) {
  setup(n);

  // Cubature constants
  cubDr  = &cubDr_g[(N - 1) * DG_CUB_NP * DG_NP];
  cubDs  = &cubDs_g[(N - 1) * DG_CUB_NP * DG_NP];
  cubV   = &cubV_g[(N - 1) * DG_CUB_NP * DG_NP];
  cubVDr = &cubVDr_g[(N - 1) * DG_CUB_NP * DG_NP];
  cubVDs = &cubVDs_g[(N - 1) * DG_CUB_NP * DG_NP];
  cubW   = &cubW_g[(N - 1) * DG_CUB_NP];
  // Grad constants
  Dr  = &Dr_g[(N - 1) * DG_NP * DG_NP];
  Drw = &Drw_g[(N - 1) * DG_NP * DG_NP];
  Ds  = &Ds_g[(N - 1) * DG_NP * DG_NP];
  Dsw = &Dsw_g[(N - 1) * DG_NP * DG_NP];
  // Gauss constants
  gaussW     = &gaussW_g[(N - 1) * DG_GF_NP];
  gF0Dr      = &gF0Dr_g[(N - 1) * DG_GF_NP * DG_NP];
  gF0Ds      = &gF0Ds_g[(N - 1) * DG_GF_NP * DG_NP];
  gF1Dr      = &gF1Dr_g[(N - 1) * DG_GF_NP * DG_NP];
  gF1Ds      = &gF1Ds_g[(N - 1) * DG_GF_NP * DG_NP];
  gF2Dr      = &gF2Dr_g[(N - 1) * DG_GF_NP * DG_NP];
  gF2Ds      = &gF2Ds_g[(N - 1) * DG_GF_NP * DG_NP];
  gFInterp0  = &gFInterp0_g[(N - 1) * DG_GF_NP * DG_NP];
  gFInterp1  = &gFInterp1_g[(N - 1) * DG_GF_NP * DG_NP];
  gFInterp2  = &gFInterp2_g[(N - 1) * DG_GF_NP * DG_NP];
  gInterp    = &gInterp_g[(N - 1) * DG_G_NP * DG_NP];

  // Other constants
  invMass = &invMass_g[(N - 1) * DG_NP * DG_NP];
  lift    = &lift_g[(N - 1) * DG_NP * 3 * DG_NPF];
  mass    = &mass_g[(N - 1) * DG_NP * DG_NP];
  v       = &v_g[(N - 1) * DG_NP * DG_NP];
  invV    = &invV_g[(N - 1) * DG_NP * DG_NP];
  r       = &r_g[(N - 1) * DG_NP];
  s       = &s_g[(N - 1) * DG_NP];
  ones    = &ones_g[(N - 1) * DG_NP];
}

DGConstants::~DGConstants() {

}

double* DGConstants::get_ptr(Constant_Matrix mat) {
  switch(mat) {
    case CUB_DR:
      return cubDr;
    case CUB_DS:
      return cubDs;
    case CUB_V:
      return cubV;
    case CUB_VDR:
      return cubVDr;
    case CUB_VDS:
      return cubVDs;
    case CUB_W:
      return cubW;
    case DR:
      return Dr;
    case DRW:
      return Drw;
    case DS:
      return Ds;
    case DSW:
      return Dsw;
    case GAUSS_W:
      return gaussW;
    case GAUSS_F0DR:
      return gF0Dr;
    case GAUSS_F0DS:
      return gF0Ds;
    case GAUSS_F1DR:
      return gF1Dr;
    case GAUSS_F1DS:
      return gF1Ds;
    case GAUSS_F2DR:
      return gF2Dr;
    case GAUSS_F2DS:
      return gF2Ds;
    case GAUSS_FINTERP0:
      return gFInterp0;
    case GAUSS_FINTERP1:
      return gFInterp1;
    case GAUSS_FINTERP2:
      return gFInterp2;
    case GAUSS_INTERP:
      return gInterp;
    case INV_MASS:
      return invMass;
    case LIFT:
      return lift;
    case MASS:
      return mass;
    case V:
      return v;
    case INV_V:
      return invV;
    case R:
      return r;
    case S:
      return s;
    case ONES:
      return ones;
  }
  return nullptr;
}
