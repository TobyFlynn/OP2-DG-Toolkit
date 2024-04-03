#ifndef __DG_CONSTANTS_H
#define __DG_CONSTANTS_H

#include "dg_compiler_defs.h"

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

class DGConstants {
public:
  enum Constant_Matrix {
    DR, DS, DT, DRW, DSW, DTW, LIFT, INV_V,
    INV_MASS, MASS, V, R, S, T,
    MM_F0, MM_F1, MM_F2, MM_F3, EMAT,

    CUB2D_R, CUB2D_S, CUB2D_W,
    CUB2D_INTERP, CUB2D_PROJ, CUB2D_PDR, CUB2D_PDS,
    CUBSURF2D_INTERP, CUBSURF2D_LIFT,

    CUB3D_R, CUB3D_S, CUB3D_T, CUB3D_W,
    CUB3D_INTERP, CUB3D_PROJ, CUB3D_PDR, CUB3D_PDS, CUB3D_PDT,
    CUBSURF3D_INTERP, CUBSURF3D_LIFT,

    INTERP_MATRIX_ARRAY
  };

  virtual void calc_interp_mats() = 0;
  virtual DG_FP* get_mat_ptr(Constant_Matrix matrix) = 0;
  virtual DG_FP* get_mat_ptr_device(Constant_Matrix matrix) = 0;
  virtual float* get_mat_ptr_device_sp(Constant_Matrix matrix) = 0;

  int N_max, Np_max, Nfp_max;
};

#endif
