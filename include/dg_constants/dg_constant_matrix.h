#pragma once

#include "dg_compiler_defs.h"

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

  bool has_multiple_orders();

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

private:
  void save_mat(DG_FP *mem_ptr, arma::mat &mat, const int N, const int max_size);

  bool multiple_orders;
  int max_rows, max_cols;
  std::vector<int> rows, cols;
  DG_FP *mat_ptr_dp;
  DG_FP *mat_ptr_dp_device;
  float *mat_ptr_sp;
  float *mat_ptr_sp_device;
};