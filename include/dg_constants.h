#ifndef __DG_CONSTANTS_H
#define __DG_CONSTANTS_H

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

class DGConstants {
public:
  enum Constant_Matrix {
    DR, DS, DT, DRW, DSW, DTW, LIFT, INV_V,

    CUB_DR, CUB_DS, CUB_V, CUB_VDR, CUB_VDS, CUB_W,

    GAUSS_W, GAUSS_F0DR, GAUSS_F0DS, GAUSS_F1DR, GAUSS_F1DS,
    GAUSS_F2DR, GAUSS_F2DS, GAUSS_FINTERP0, GAUSS_FINTERP1,
    GAUSS_FINTERP2, GAUSS_INTERP,

    INV_MASS_GAUSS_INTERP_T,

    INV_MASS, MASS, V, R, S,

    MM_F0, MM_F1, MM_F2, MM_F3,

    INTERP_MATRIX_ARRAY
  };

  virtual void calc_interp_mats() = 0;
  virtual double* get_mat_ptr(Constant_Matrix matrix) = 0;
};

#endif