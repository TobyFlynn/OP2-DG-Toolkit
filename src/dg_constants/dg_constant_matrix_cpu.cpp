#include "dg_constants/dg_constant_matrix.h"

#include "dg_op2_custom_blas.h"

DGConstantMatrix::DGConstantMatrix(const int _max_rows, const int _max_cols, bool _multiple_orders) :
                            max_rows(_max_rows), max_cols(_max_cols), multiple_orders(_multiple_orders) {
  if(multiple_orders) {
    mat_ptr_dp = (DG_FP *)calloc(DG_ORDER * max_rows * max_cols, sizeof(DG_FP));
    mat_ptr_sp = (float *)calloc(DG_ORDER * max_rows * max_cols, sizeof(float));
    for(int i = 0; i < DG_ORDER; i++) {
      rows.push_back(max_rows);
      cols.push_back(max_cols);
    }
  } else {
    mat_ptr_dp = (DG_FP *)calloc(max_rows * max_cols, sizeof(DG_FP));
    mat_ptr_sp = (float *)calloc(max_rows * max_cols, sizeof(float));
  }
}

DGConstantMatrix::~DGConstantMatrix() {
  free(mat_ptr_dp);
  free(mat_ptr_sp);
}

void DGConstantMatrix::transfer_to_device() {
  mat_ptr_dp_device = mat_ptr_dp;
  mat_ptr_sp_device = mat_ptr_sp;
}

void DGConstantMatrix::profile_blas(DGMesh *mesh) {
  if(multiple_orders) {
    for(int i = 0; i < DG_ORDER; i++) {
      use_custom_kernel_dp.push_back(op2_gemv_have_dp_custom_kernel(rows[i], cols[i]));
      use_custom_kernel_sp.push_back(op2_gemv_have_sp_custom_kernel(rows[i], cols[i]));
    }
  } else {
    use_custom_kernel_dp.push_back(op2_gemv_have_dp_custom_kernel(max_rows, max_cols));
    use_custom_kernel_sp.push_back(op2_gemv_have_sp_custom_kernel(max_rows, max_cols));
  }
}
