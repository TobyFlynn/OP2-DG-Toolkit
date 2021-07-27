#ifndef __DG_CONSTANTS_H
#define __DG_CONSTANTS_H

#include "dg_global_constants.h"

#ifdef OP2_DG_CUDA
#include "cublas_v2.h"
#endif

class DGConstants {
public:
  enum Constant_Matrix {
    CUB_DR, CUB_DS, CUB_V, CUB_VDR, CUB_VDS, CUB_W,

    DR, DRW, DS, DSW,

    GAUSS_W, GAUSS_F0DR, GAUSS_F0DR_R, GAUSS_F0DS, GAUSS_F0DS_R,
    GAUSS_F1DR, GAUSS_F1DR_R, GAUSS_F1DS, GAUSS_F1DS_R,
    GAUSS_F2DR, GAUSS_F2DR_R, GAUSS_F2DS, GAUSS_F2DS_R,
    GAUSS_FINTERP0, GAUSS_FINTERP0_R, GAUSS_FINTERP1, GAUSS_FINTERP1_R,
    GAUSS_FINTERP2, GAUSS_FINTERP2_R, GAUSS_INTERP,

    INV_MASS, LIFT, MASS, R, S, ONES
  };

  DGConstants();
  ~DGConstants();

  double* get_ptr(Constant_Matrix mat);

  double *cubDr, *cubDr_d;
  double *cubDs, *cubDs_d;
  double *cubV, *cubV_d;
  double *cubVDr, *cubVDr_d;
  double *cubVDs, *cubVDs_d;
  double *cubW, *cubW_d;

  double *Dr, *Dr_d;
  double *Drw, *Drw_d;
  double *Ds, *Ds_d;
  double *Dsw, *Dsw_d;

  double *gaussW, *gaussW_d;
  double *gF0Dr, *gF0Dr_d;
  double *gF0DrR, *gF0DrR_d;
  double *gF0Ds, *gF0Ds_d;
  double *gF0DsR, *gF0DsR_d;
  double *gF1Dr, *gF1Dr_d;
  double *gF1DrR, *gF1DrR_d;
  double *gF1Ds, *gF1Ds_d;
  double *gF1DsR, *gF1DsR_d;
  double *gF2Dr, *gF2Dr_d;
  double *gF2DrR, *gF2DrR_d;
  double *gF2Ds, *gF2Ds_d;
  double *gF2DsR, *gF2DsR_d;
  double *gFInterp0, *gFInterp0_d;
  double *gFInterp0R, *gFInterp0R_d;
  double *gFInterp1, *gFInterp1_d;
  double *gFInterp1R, *gFInterp1R_d;
  double *gFInterp2, *gFInterp2_d;
  double *gFInterp2R, *gFInterp2R_d;
  double *gInterp, *gInterp_d;

  double *invMass, *invMass_d;
  double *lift, *lift_d;
  double *mass, *mass_d;
  double *r, *r_d;
  double *s, *s_d;
  double *ones, *ones_d;
  #ifdef OP2_DG_CUDA
  cublasHandle_t handle;
  #endif
};

#endif
