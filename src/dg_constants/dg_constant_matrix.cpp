#include "dg_constants/dg_constant_matrix.h"

void DGConstantMatrix::save_mat(DG_FP *mem_ptr, arma::mat &mat, const int N, const int max_size) {
  #ifdef DG_COL_MAJ
  arma::Mat<DG_FP> mat_2 = arma::conv_to<arma::Mat<DG_FP>>::from(mat);
  #else
  arma::Mat<DG_FP> mat_2 = arma::conv_to<arma::Mat<DG_FP>>::from(mat.t());
  #endif
  memcpy(&mem_ptr[(N - 1) * max_size], mat_2.memptr(), mat_2.n_elem * sizeof(DG_FP));
}

void DGConstantMatrix::set_mat(arma::mat &matrix) {
  save_mat(mat_ptr_dp, matrix, 1, max_rows * max_cols);

  for(int i = 0; i < max_rows * max_cols; i++) {
    mat_ptr_sp[i] = static_cast<float>(mat_ptr_dp[i]);
  }
}

void DGConstantMatrix::set_mat(arma::mat &matrix, const int N) {
  rows[N - 1] = matrix.n_rows;
  cols[N - 1] = matrix.n_cols;

  save_mat(mat_ptr_dp, matrix, N, max_rows * max_cols);

  for(int i = 0; i < rows[N - 1] * cols[N - 1]; i++) {
    mat_ptr_sp[(N - 1) * max_rows * max_cols + i] = static_cast<float>(mat_ptr_dp[(N - 1) * max_rows * max_cols + i]);
  }
}

int DGConstantMatrix::get_rows(const int N) {
  if(multiple_orders)
    return rows[N - 1];
  else
    return max_rows;
}

int DGConstantMatrix::get_cols(const int N) {
  if(multiple_orders)
    return cols[N - 1];
  else
    return max_cols;
}

int DGConstantMatrix::get_max_rows() {
  return max_rows;
}

int DGConstantMatrix::get_max_cols() {
  return max_cols;
}

DG_FP* DGConstantMatrix::get_mat_ptr_dp() {
  return mat_ptr_dp;
}

DG_FP* DGConstantMatrix::get_mat_ptr_dp_device() {
  return mat_ptr_dp_device;
}

float* DGConstantMatrix::get_mat_ptr_sp() {
  return mat_ptr_sp;
}

float* DGConstantMatrix::get_mat_ptr_sp_device() {
  return mat_ptr_sp_device;
}

DG_FP* DGConstantMatrix::get_mat_ptr_dp(const int N) {
  if(multiple_orders)
    return mat_ptr_dp + (N - 1) * max_rows * max_cols;
  else
    return mat_ptr_dp;
}

DG_FP* DGConstantMatrix::get_mat_ptr_dp_device(const int N) {
  if(multiple_orders)
    return mat_ptr_dp_device + (N - 1) * max_rows * max_cols;
  else
    return mat_ptr_dp_device;
}

float* DGConstantMatrix::get_mat_ptr_sp(const int N) {
  if(multiple_orders)
    return mat_ptr_sp + (N - 1) * max_rows * max_cols;
  else
    return mat_ptr_sp;
}

float* DGConstantMatrix::get_mat_ptr_sp_device(const int N) {
  if(multiple_orders)
    return mat_ptr_sp_device + (N - 1) * max_rows * max_cols;
  else
    return mat_ptr_sp_device;
}

bool DGConstantMatrix::has_multiple_orders() {
  return multiple_orders;
}

bool DGConstantMatrix::use_custom_blas_kernel_dp(const int N) {
  if(multiple_orders) {
    return use_custom_kernel_dp[N - 1];
  } else {
    return use_custom_kernel_dp[0];
  }
}

bool DGConstantMatrix::use_custom_blas_kernel_sp(const int N) {
  if(multiple_orders) {
    return use_custom_kernel_sp[N - 1];
  } else {
    return use_custom_kernel_sp[0];
  }
}
