#include "dg_constants/dg_constant_matrix.h"

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