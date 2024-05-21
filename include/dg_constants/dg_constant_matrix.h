#pragma once

#include "dg_compiler_defs.h"
#include "dg_mesh/dg_mesh.h"

#include <vector>

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

class DGConstantMatrix {
public:
  DGConstantMatrix(const int _max_rows, const int _max_cols, bool _multiple_orders);
  ~DGConstantMatrix();

  void set_mat(arma::mat &matrix);
  void set_mat(arma::mat &matrix, const int N);
  void transfer_to_device();
  void profile_blas(DGMesh *mesh);

  bool has_multiple_orders();
  bool use_custom_blas_kernel_dp(const int N);
  bool use_custom_blas_kernel_sp(const int N);

  int get_rows(const int N);
  int get_cols(const int N);
  int get_max_rows();
  int get_max_cols();

  DG_FP* get_mat_ptr_dp();
  DG_FP* get_mat_ptr_dp_device();
  float* get_mat_ptr_sp();
  float* get_mat_ptr_sp_device();

  DG_FP* get_mat_ptr_dp(const int N);
  DG_FP* get_mat_ptr_dp_device(const int N);
  float* get_mat_ptr_sp(const int N);
  float* get_mat_ptr_sp_device(const int N);

  // So that I can copy the address of these pointers with cudaMemcpyToSymbol
  DG_FP *mat_ptr_dp;
  DG_FP *mat_ptr_dp_device;
  float *mat_ptr_sp;
  float *mat_ptr_sp_device;

private:
  void save_mat(DG_FP *mem_ptr, arma::mat &mat, const int N, const int max_size);

  bool multiple_orders;
  int max_rows, max_cols;
  std::vector<int> rows, cols;
  std::vector<bool> use_custom_kernel_dp, use_custom_kernel_sp;
};
