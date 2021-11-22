#include "dg_constants.h"

#include "dg_global_constants.h"

DGConstants::DGConstants() {
  setup(DG_ORDER);

  // Cubature constants
  cubDr  = cubDr_g;
  cubDs  = cubDs_g;
  cubV   = cubV_g;
  cubVDr = cubVDr_g;
  cubVDs = cubVDs_g;
  cubW   = cubW_g;
  // Grad constants
  Dr  = Dr_g;
  Drw = Drw_g;
  Ds  = Ds_g;
  Dsw = Dsw_g;
  // Gauss constants
  gaussW     = gaussW_g;
  gF0Dr      = gF0Dr_g;
  gF0Ds      = gF0Ds_g;
  gF1Dr      = gF1Dr_g;
  gF1Ds      = gF1Ds_g;
  gF2Dr      = gF2Dr_g;
  gF2Ds      = gF2Ds_g;
  gFInterp0  = gFInterp0_g;
  gFInterp1  = gFInterp1_g;
  gFInterp2  = gFInterp2_g;
  gInterp    = gInterp_g;

  // Other constants
  invMass = invMass_g;
  lift    = lift_g;
  mass    = mass_g;
  v       = v_g;
  invV    = invV_g;
  r       = r_g;
  s       = s_g;
  ones    = ones_g;
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
