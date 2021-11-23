#ifndef __DG_CONSTANTS_H
#define __DG_CONSTANTS_H

#include "dg_global_constants.h"

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

class DGConstants {
public:
  enum Constant_Matrix {
    CUB_DR, CUB_DS, CUB_V, CUB_VDR, CUB_VDS, CUB_W,

    DR, DRW, DS, DSW,

    GAUSS_W, GAUSS_F0DR, GAUSS_F0DS, GAUSS_F1DR, GAUSS_F1DS,
    GAUSS_F2DR, GAUSS_F2DS, GAUSS_FINTERP0, GAUSS_FINTERP1,
    GAUSS_FINTERP2, GAUSS_INTERP,

    INV_MASS, LIFT, MASS, INV_V, V, R, S, ONES
  };

  DGConstants(const int n);
  ~DGConstants();

  void setup(const int n);
  void cubature(const int nCub);
  void gauss(const int nGauss);

  int N, Nfp, Np, cNp, gNp, gNfp;

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
  double *gF0Ds, *gF0Ds_d;
  double *gF1Dr, *gF1Dr_d;
  double *gF1Ds, *gF1Ds_d;
  double *gF2Dr, *gF2Dr_d;
  double *gF2Ds, *gF2Ds_d;
  double *gFInterp0, *gFInterp0_d;
  double *gFInterp1, *gFInterp1_d;
  double *gFInterp2, *gFInterp2_d;
  double *gInterp, *gInterp_d;

  double *invMass, *invMass_d;
  double *lift, *lift_d;
  double *mass, *mass_d;
  double *v, *v_d;
  double *invV, *invV_d;
  double *r, *r_d;
  double *s, *s_d;
  double *ones, *ones_d;

private:
  arma::vec x_, y_, r_, s_;
  arma::uvec fmask1_, fmask2_, fmask3_, fmask_;
  arma::mat V_, invV_, MassMatrix_, Dr_, Ds_, lift_, Drw_, Dsw_;

  arma::vec cub_w_;
  arma::mat cub_V_, cub_Dr_, cub_Ds_;

  arma::vec gauss_w_;
  arma::mat gauss_interp_, gauss_interp1_, gauss_interp2_, gauss_interp3_;
};

#endif
