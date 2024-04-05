#ifndef __DG_CONSTANTS_3D_H
#define __DG_CONSTANTS_3D_H

#include "dg_compiler_defs.h"

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "dg_constants.h"
#include "dg_constant_matrix.h"

class DGConstants3D : public DGConstants {
public:
  DGConstants3D(const int n_);
  ~DGConstants3D();

  void calc_interp_mats() override;
  DG_FP* get_mat_ptr(Constant_Matrix matrix) override;
  float* get_mat_ptr_sp(Constant_Matrix matrix) override;
  DG_FP* get_mat_ptr_device(Constant_Matrix matrix) override;
  float* get_mat_ptr_device_sp(Constant_Matrix matrix) override;

private:
  void getCubatureData(const int N, arma::vec &cubr, arma::vec &cubs, arma::vec &cubt, arma::vec &cubw);
  void transfer_kernel_ptrs();
  void clean_up_kernel_ptrs();

  // Map of DG constant matrices
  std::map<int,DGConstantMatrix*> dg_mats;

  // Pointers to all vectors that are returned by get_mat_ptr
  DG_FP *r_ptr, *s_ptr, *t_ptr;
  DG_FP *cub_r_ptr, *cub_s_ptr, *cub_t_ptr, *cub_w_ptr;

  // Effectively a 2D array of interp matrices. Size [DG_ORDER][DG_ORDER]
  // To get an interp array:
  // int ind = ((order_from - 1) * DG_ORDER + (order_to - 1)) * DG_NP * DG_NP
  DG_FP *order_interp_ptr;
  float *order_interp_ptr_sp;
};

#endif
