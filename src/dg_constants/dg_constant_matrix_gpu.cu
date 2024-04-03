#include "dg_constants/dg_constant_matrix.h"

#include "op_cuda_rt_support.h"

DGConstantMatrix::DGConstantMatrix(const int _max_rows, const int _max_cols, bool _multiple_orders) : 
                            max_rows(_max_rows), max_cols(_max_cols), multiple_orders(_multiple_orders) {
  if(multiple_orders) {
    mat_ptr_dp = (DG_FP *)calloc(DG_ORDER * max_rows * max_cols, sizeof(DG_FP));
    mat_ptr_sp = (float *)calloc(DG_ORDER * max_rows * max_cols, sizeof(float));
    cutilSafeCall(cudaMalloc(&mat_ptr_dp_device, DG_ORDER * max_rows * max_cols * sizeof(DG_FP)));
    cutilSafeCall(cudaMalloc(&mat_ptr_sp_device, DG_ORDER * max_rows * max_cols * sizeof(float)));
    for(int i = 0; i < DG_ORDER; i++) {
      rows.push_back(max_rows);
      cols.push_back(max_cols);
    }
  } else {
    mat_ptr_dp = (DG_FP *)calloc(max_rows * max_cols, sizeof(DG_FP));
    mat_ptr_sp = (float *)calloc(max_rows * max_cols, sizeof(float));
    cutilSafeCall(cudaMalloc(&mat_ptr_dp_device, max_rows * max_cols * sizeof(DG_FP)));
    cutilSafeCall(cudaMalloc(&mat_ptr_sp_device, max_rows * max_cols * sizeof(float)));
  }
}

DGConstantMatrix::~DGConstantMatrix() {
  free(mat_ptr_dp);
  free(mat_ptr_sp);
  cudaFree(mat_ptr_dp_device);
  cudaFree(mat_ptr_sp_device);
}

void DGConstantMatrix::transfer_to_device() {
  if(multiple_orders) {
    cutilSafeCall(cudaMemcpy(mat_ptr_dp_device, mat_ptr_dp, DG_ORDER * max_rows * max_cols * sizeof(DG_FP), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(mat_ptr_sp_device, mat_ptr_sp, DG_ORDER * max_rows * max_cols * sizeof(float), cudaMemcpyHostToDevice));
  } else {
    cutilSafeCall(cudaMemcpy(mat_ptr_dp_device, mat_ptr_dp, max_rows * max_cols * sizeof(DG_FP), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(mat_ptr_sp_device, mat_ptr_sp, max_rows * max_cols * sizeof(float), cudaMemcpyHostToDevice));
  }
}