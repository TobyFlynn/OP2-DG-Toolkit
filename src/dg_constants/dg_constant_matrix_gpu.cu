#include "dg_constants/dg_constant_matrix.h"

#include "op_cuda_rt_support.h"

#include "dg_op2_custom_blas.h"

#include <sys/time.h>
#include <vector>
#include <numeric>
#include <algorithm>

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

inline double get_time() {
  struct timeval t;
  gettimeofday(&t, (struct timezone *)0);
  return t.tv_sec + t.tv_usec * 1.0e-6;
}

void DGConstantMatrix::profile_blas(DGMesh *mesh) {
  if(multiple_orders) {
    for(int order = 0; order < DG_ORDER; order++) {
      // Check for double precision custom kernel
      if(op2_gemv_have_dp_custom_kernel(get_rows(order + 1), get_cols(order + 1))) {
        // Create temp dats used in the profiling
        op_dat dat_x = op_decl_dat_temp(mesh->cells, max_cols, DG_FP_STR, (DG_FP *)NULL, "tmp_x");
        op_dat dat_y = op_decl_dat_temp(mesh->cells, max_rows, DG_FP_STR, (DG_FP *)NULL, "tmp_y");
        std::vector<double> custom_kernel_times, standard_blas_times;
        const DG_FP alpha = 1.0;
        const DG_FP beta = 0.0;
        bool transpose = false;
        // 5 trials of each BLAS method
        for(int i = 0; i < 5; i++) {
          const double start_c = get_time();
          custom_kernel_gemv(mesh->cells, transpose, get_rows(order + 1), get_cols(order + 1), alpha, beta, get_mat_ptr_dp_device(order + 1), dat_x, dat_y);
          const double end_c = get_time();
          custom_kernel_times.push_back(end_c - start_c);

          const double start_e = get_time();
          standard_blas_lib_gemv(mesh->cells, transpose, get_rows(order + 1), get_cols(order + 1), alpha, beta, get_mat_ptr_dp_device(order + 1), dat_x, dat_y);
          const double end_e = get_time();
          standard_blas_times.push_back(end_e - start_e);
        }
        // Discard longest 2 and average the remaining 3
        auto max_it = std::max_element(custom_kernel_times.begin(), custom_kernel_times.end());
        custom_kernel_times.erase(max_it);
        max_it = std::max_element(custom_kernel_times.begin(), custom_kernel_times.end());
        custom_kernel_times.erase(max_it);
        max_it = std::max_element(standard_blas_times.begin(), standard_blas_times.end());
        standard_blas_times.erase(max_it);
        max_it = std::max_element(standard_blas_times.begin(), standard_blas_times.end());
        standard_blas_times.erase(max_it);

        const double avg_custom = std::accumulate(custom_kernel_times.begin(), custom_kernel_times.end(), 0.0) / 3.0;
        const double avg_standard = std::accumulate(standard_blas_times.begin(), standard_blas_times.end(), 0.0) / 3.0;

        // Decide whether to use custom or standard BLAS
        if(avg_custom < avg_standard) {
          use_custom_kernel_dp.push_back(true);
          // op_printf("Using custom, %g vs %g\n", avg_custom, avg_standard);
        } else {
          use_custom_kernel_dp.push_back(false);
          // op_printf("Using standard, %g vs %g\n", avg_custom, avg_standard);
        }

        op_free_dat_temp(dat_x);
        op_free_dat_temp(dat_y);
      } else {
        use_custom_kernel_dp.push_back(false);
      }

      // Check for single precision custom kernel
      if(op2_gemv_have_sp_custom_kernel(get_rows(order + 1), get_cols(order + 1))) {
        // Create temp dats used in the profiling
        op_dat dat_x = op_decl_dat_temp(mesh->cells, max_cols, "float", (float *)NULL, "tmp_x");
        op_dat dat_y = op_decl_dat_temp(mesh->cells, max_rows, "float", (float *)NULL, "tmp_y");
        std::vector<double> custom_kernel_times, standard_blas_times;
        const DG_FP alpha = 1.0;
        const DG_FP beta = 0.0;
        bool transpose = false;
        // 5 trials of each BLAS method
        for(int i = 0; i < 5; i++) {
          const double start_c = get_time();
          custom_kernel_gemv_sp(mesh->cells, transpose, get_rows(order + 1), get_cols(order + 1), alpha, beta, get_mat_ptr_sp_device(order + 1), dat_x, dat_y);
          const double end_c = get_time();
          custom_kernel_times.push_back(end_c - start_c);

          const double start_e = get_time();
          standard_blas_lib_gemv_sp(mesh->cells, transpose, get_rows(order + 1), get_cols(order + 1), alpha, beta, get_mat_ptr_sp_device(order + 1), dat_x, dat_y);
          const double end_e = get_time();
          standard_blas_times.push_back(end_e - start_e);
        }
        // Discard longest 2 and average the remaining 3
        auto max_it = std::max_element(custom_kernel_times.begin(), custom_kernel_times.end());
        custom_kernel_times.erase(max_it);
        max_it = std::max_element(custom_kernel_times.begin(), custom_kernel_times.end());
        custom_kernel_times.erase(max_it);
        max_it = std::max_element(standard_blas_times.begin(), standard_blas_times.end());
        standard_blas_times.erase(max_it);
        max_it = std::max_element(standard_blas_times.begin(), standard_blas_times.end());
        standard_blas_times.erase(max_it);

        const double avg_custom = std::accumulate(custom_kernel_times.begin(), custom_kernel_times.end(), 0.0) / 3.0;
        const double avg_standard = std::accumulate(standard_blas_times.begin(), standard_blas_times.end(), 0.0) / 3.0;

        // Decide whether to use custom or standard BLAS
        if(avg_custom < avg_standard) {
          use_custom_kernel_sp.push_back(true);
          // op_printf("Using custom, %g vs %g\n", avg_custom, avg_standard);
        } else {
          use_custom_kernel_sp.push_back(false);
          // op_printf("Using standard, %g vs %g\n", avg_custom, avg_standard);
        }

        op_free_dat_temp(dat_x);
        op_free_dat_temp(dat_y);
      } else {
        use_custom_kernel_sp.push_back(false);
      }
    }
  } else {
    // Check for double precision custom kernel
    if(op2_gemv_have_dp_custom_kernel(max_rows, max_cols)) {
      // Create temp dats used in the profiling
      op_dat dat_x = op_decl_dat_temp(mesh->cells, max_cols, DG_FP_STR, (DG_FP *)NULL, "tmp_x");
      op_dat dat_y = op_decl_dat_temp(mesh->cells, max_rows, DG_FP_STR, (DG_FP *)NULL, "tmp_y");
      std::vector<double> custom_kernel_times, standard_blas_times;
      const DG_FP alpha = 1.0;
      const DG_FP beta = 0.0;
      bool transpose = false;
      // 5 trials of each BLAS method
      for(int i = 0; i < 5; i++) {
        const double start_c = get_time();
        custom_kernel_gemv(mesh->cells, transpose, max_rows, max_cols, alpha, beta, get_mat_ptr_dp_device(DG_ORDER), dat_x, dat_y);
        const double end_c = get_time();
        custom_kernel_times.push_back(end_c - start_c);

        const double start_e = get_time();
        standard_blas_lib_gemv(mesh->cells, transpose, max_rows, max_cols, alpha, beta, get_mat_ptr_dp_device(DG_ORDER), dat_x, dat_y);
        const double end_e = get_time();
        standard_blas_times.push_back(end_e - start_e);
      }
      // Discard longest 2 and average the remaining 3
      auto max_it = std::max_element(custom_kernel_times.begin(), custom_kernel_times.end());
      custom_kernel_times.erase(max_it);
      max_it = std::max_element(custom_kernel_times.begin(), custom_kernel_times.end());
      custom_kernel_times.erase(max_it);
      max_it = std::max_element(standard_blas_times.begin(), standard_blas_times.end());
      standard_blas_times.erase(max_it);
      max_it = std::max_element(standard_blas_times.begin(), standard_blas_times.end());
      standard_blas_times.erase(max_it);

      const double avg_custom = std::accumulate(custom_kernel_times.begin(), custom_kernel_times.end(), 0.0) / 3.0;
      const double avg_standard = std::accumulate(standard_blas_times.begin(), standard_blas_times.end(), 0.0) / 3.0;

      // Decide whether to use custom or standard BLAS
      if(avg_custom < avg_standard) {
        use_custom_kernel_dp.push_back(true);
        // op_printf("Using custom, %g vs %g\n", avg_custom, avg_standard);
      } else {
        use_custom_kernel_dp.push_back(false);
        // op_printf("Using standard, %g vs %g\n", avg_custom, avg_standard);
      }

      op_free_dat_temp(dat_x);
      op_free_dat_temp(dat_y);
    } else {
      use_custom_kernel_dp.push_back(false);
    }

    // Check for single precision custom kernel
    if(op2_gemv_have_sp_custom_kernel(max_rows, max_cols)) {
      // Create temp dats used in the profiling
      op_dat dat_x = op_decl_dat_temp(mesh->cells, max_cols, "float", (float *)NULL, "tmp_x");
      op_dat dat_y = op_decl_dat_temp(mesh->cells, max_rows, "float", (float *)NULL, "tmp_y");
      std::vector<double> custom_kernel_times, standard_blas_times;
      const DG_FP alpha = 1.0;
      const DG_FP beta = 0.0;
      bool transpose = false;
      // 5 trials of each BLAS method
      for(int i = 0; i < 5; i++) {
        const double start_c = get_time();
        custom_kernel_gemv_sp(mesh->cells, transpose, max_rows, max_cols, alpha, beta, get_mat_ptr_sp_device(DG_ORDER), dat_x, dat_y);
        const double end_c = get_time();
        custom_kernel_times.push_back(end_c - start_c);

        const double start_e = get_time();
        standard_blas_lib_gemv_sp(mesh->cells, transpose, max_rows, max_cols, alpha, beta, get_mat_ptr_sp_device(DG_ORDER), dat_x, dat_y);
        const double end_e = get_time();
        standard_blas_times.push_back(end_e - start_e);
      }
      // Discard longest 2 and average the remaining 3
      auto max_it = std::max_element(custom_kernel_times.begin(), custom_kernel_times.end());
      custom_kernel_times.erase(max_it);
      max_it = std::max_element(custom_kernel_times.begin(), custom_kernel_times.end());
      custom_kernel_times.erase(max_it);
      max_it = std::max_element(standard_blas_times.begin(), standard_blas_times.end());
      standard_blas_times.erase(max_it);
      max_it = std::max_element(standard_blas_times.begin(), standard_blas_times.end());
      standard_blas_times.erase(max_it);

      const double avg_custom = std::accumulate(custom_kernel_times.begin(), custom_kernel_times.end(), 0.0) / 3.0;
      const double avg_standard = std::accumulate(standard_blas_times.begin(), standard_blas_times.end(), 0.0) / 3.0;

      // Decide whether to use custom or standard BLAS
      if(avg_custom < avg_standard) {
        use_custom_kernel_sp.push_back(true);
        // op_printf("Using custom, %g vs %g\n", avg_custom, avg_standard);
      } else {
        use_custom_kernel_sp.push_back(false);
        // op_printf("Using standard, %g vs %g\n", avg_custom, avg_standard);
      }

      op_free_dat_temp(dat_x);
      op_free_dat_temp(dat_y);
    } else {
      use_custom_kernel_sp.push_back(false);
    }
  }
}
