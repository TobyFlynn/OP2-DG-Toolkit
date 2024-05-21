#ifndef __DG_CONSTANTS_2D_H
#define __DG_CONSTANTS_2D_H

#include "dg_compiler_defs.h"

#include <map>

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "dg_constants.h"
#include "dg_constant_matrix.h"

class DGConstants2D : public DGConstants {
public:
  DGConstants2D(const int n_);
  ~DGConstants2D();

  void calc_interp_mats() override;
  void profile_blas(DGMesh *mesh) override;
  DG_FP* get_mat_ptr(Constant_Matrix matrix) override;
  float* get_mat_ptr_sp(Constant_Matrix matrix) override;
  DG_FP* get_mat_ptr_device(Constant_Matrix matrix) override;
  float* get_mat_ptr_device_sp(Constant_Matrix matrix) override;
  DGConstantMatrix* get_dg_constant_matrix_ptr(Constant_Matrix matrix) override;

  int cNp_max, gNp_max, gNfp_max;

private:
  void cubature(const int nCub);
  void gauss(const int nGauss);

  void transfer_kernel_ptrs();
  void clean_up_kernel_ptrs();

  // Map of DG constant matrices
  std::map<int,DGConstantMatrix*> dg_mats;

  // Pointers to all vectors that are returned by get_mat_ptr
  DG_FP *r_ptr, *s_ptr;
  DG_FP *cub_r_ptr, *cub_s_ptr, *cub_w_ptr;

  // Effectively a 2D array of interp matrices. Size [DG_ORDER][DG_ORDER]
  // To get an interp array:
  // int ind = ((order_from - 1) * DG_ORDER + (order_to - 1)) * DG_NP * DG_NP
  DG_FP *order_interp_ptr;
  float *order_interp_ptr_sp;
};

#endif
