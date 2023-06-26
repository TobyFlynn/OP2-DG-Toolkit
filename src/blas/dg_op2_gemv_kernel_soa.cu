#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

#include "dg_compiler_defs.h"
#include "dg_mesh/dg_mesh.h"

#include "kernels/non_templated_soa.h"
#include "kernels/templated_soa.h"

void custom_kernel_gemv(op_set set, const bool t, const int m, const int n, const DG_FP alpha,
  const DG_FP beta, const DG_FP *matrix, op_dat x, op_dat y) {

  int nargs = 2;
  op_arg args[2] = {
    op_arg_dat(x, -1, OP_ID, x->dim, DG_FP_STR, OP_READ),
    op_arg_dat(y, -1, OP_ID, y->dim, DG_FP_STR, OP_RW)
  };

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2, 0);
  if (set_size > 0) {
    //set CUDA execution parameters
    int nthread = 256;
    const int nblocks = set->size / nthread + 1;
    const int strideX = getSetSizeFromOpArg(&args[0]);
    const int strideY = getSetSizeFromOpArg(&args[1]);

    if(t) {
      switch(m) {
        // The number of nodes for each order
        case 4:
          templated_cuda_gemm_T_gpu<4><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                              strideX, strideY, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 10:
          templated_cuda_gemm_T_gpu<10><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                              strideX, strideY, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 20:
          templated_cuda_gemm_T_gpu<20><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                              strideX, strideY, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        // The number of face nodes for each order
        case 12:
          templated_cuda_gemm_T_gpu<12><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                              strideX, strideY, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 24:
          templated_cuda_gemm_T_gpu<24><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                              strideX, strideY, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 40:
          templated_cuda_gemm_T_gpu<40><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                              strideX, strideY, alpha, beta,
                                              matrix, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        default:
          cuda_gemm_T_gpu<<<nblocks,nthread,m*n*sizeof(double)>>>(m, n, strideX, strideY, alpha, beta,
                                               matrix, (double *) args[0].data_d,
                                               (double *) args[1].data_d, set->size);
      }
    } else {
      if(m == 4 && n == 4) {
        templated_cuda_gemm_gpu<4,4><<<nblocks,nthread,m*sizeof(double)>>>(
                                            strideX, strideY, alpha, beta,
                                            matrix, (double *) args[0].data_d,
                                            (double *) args[1].data_d, set->size);
      } else if(m == 10 && n == 10) {
        templated_cuda_gemm_gpu<10,10><<<nblocks,nthread,m*sizeof(double)>>>(
                                            strideX, strideY, alpha, beta,
                                            matrix, (double *) args[0].data_d,
                                            (double *) args[1].data_d, set->size);
      } else if(m == 20 && n == 20) {
        templated_cuda_gemm_gpu<20,20><<<nblocks,nthread,m*sizeof(double)>>>(
                                            strideX, strideY, alpha, beta,
                                            matrix, (double *) args[0].data_d,
                                            (double *) args[1].data_d, set->size);
      } else if(m == 4 && n == 12) {
        templated_cuda_gemm_gpu<4,12><<<nblocks,nthread,m*sizeof(double)>>>(
                                            strideX, strideY, alpha, beta,
                                            matrix, (double *) args[0].data_d,
                                            (double *) args[1].data_d, set->size);
      } else if(m == 10 && n == 24) {
        templated_cuda_gemm_gpu<10,24><<<nblocks,nthread,m*sizeof(double)>>>(
                                            strideX, strideY, alpha, beta,
                                            matrix, (double *) args[0].data_d,
                                            (double *) args[1].data_d, set->size);
      } else if(m == 20 && n == 40) {
        templated_cuda_gemm_gpu<20,40><<<nblocks,nthread,m*sizeof(double)>>>(
                                            strideX, strideY, alpha, beta,
                                            matrix, (double *) args[0].data_d,
                                            (double *) args[1].data_d, set->size);
      } else {
        cuda_gemm_gpu<<<nblocks,nthread,m*n*sizeof(double)>>>(m, n, strideX, strideY, alpha, beta,
                                           matrix, (double *) args[0].data_d,
                                           (double *) args[1].data_d, set->size);
      }
    }
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
}

void custom_kernel_gemv_halo_exchange(op_set set, const bool t, const int m, const int n, const DG_FP alpha,
  const DG_FP beta, const DG_FP *matrix, op_dat x, op_dat y) {

  int nargs = 2;
  op_arg args[2] = {
    op_arg_dat(x, -1, OP_ID, x->dim, DG_FP_STR, OP_READ),
    op_arg_dat(y, -1, OP_ID, y->dim, DG_FP_STR, beta == 0.0 ? OP_WRITE : OP_RW)
  };

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2, 1);
  if (set_size > 0) {
    //set CUDA execution parameters
    const int strideX = getSetSizeFromOpArg(&args[0]);
    const int strideY = getSetSizeFromOpArg(&args[1]);

    for ( int round=0; round<2; round++ ){
      if (round==1) {
        op_mpi_wait_all_grouped(nargs, args, 2, 1);
      }

      int start = round==0 ? 0 : set->core_size;
      int end = round==0 ? set->core_size : set->size + set->exec_size + set->nonexec_size;
      if(end - start <= 0) continue;

      const double *x_ptr = (double *)args[0].data_d + start;
      double *y_ptr = (double *)args[1].data_d + start;

      const int nthread = 256;
      const int nblocks = (end - start) / nthread + 1;

      if(t) {
        switch(m) {
          // The number of nodes for each order
          case 4:
            templated_cuda_gemm_T_gpu<4><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                                strideX, strideY, alpha, beta,
                                                matrix, x_ptr, y_ptr, end - start);
            break;
          case 10:
            templated_cuda_gemm_T_gpu<10><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                                strideX, strideY, alpha, beta,
                                                matrix, x_ptr, y_ptr, end - start);
            break;
          case 20:
            templated_cuda_gemm_T_gpu<20><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                                strideX, strideY, alpha, beta,
                                                matrix, x_ptr, y_ptr, end - start);
            break;
          // The number of face nodes for each order
          case 12:
            templated_cuda_gemm_T_gpu<12><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                                strideX, strideY, alpha, beta,
                                                matrix, x_ptr, y_ptr, end - start);
            break;
          case 24:
            templated_cuda_gemm_T_gpu<24><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                                strideX, strideY, alpha, beta,
                                                matrix, x_ptr, y_ptr, end - start);
            break;
          case 40:
            templated_cuda_gemm_T_gpu<40><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                                strideX, strideY, alpha, beta,
                                                matrix, x_ptr, y_ptr, end - start);
            break;
          default:
            cuda_gemm_T_gpu<<<nblocks,nthread,m*n*sizeof(double)>>>(m, n, strideX, strideY, alpha, beta,
                                                 matrix, x_ptr, y_ptr, end - start);
        }
      } else {
        if(m == 4 && n == 4) {
          templated_cuda_gemm_gpu<4,4><<<nblocks,nthread,m*sizeof(double)>>>(
                                              strideX, strideY, alpha, beta,
                                              matrix, x_ptr, y_ptr, end - start);
        } else if(m == 10 && n == 10) {
          templated_cuda_gemm_gpu<10,10><<<nblocks,nthread,m*sizeof(double)>>>(
                                              strideX, strideY, alpha, beta,
                                              matrix, x_ptr, y_ptr, end - start);
        } else if(m == 20 && n == 20) {
          templated_cuda_gemm_gpu<20,20><<<nblocks,nthread,m*sizeof(double)>>>(
                                              strideX, strideY, alpha, beta,
                                              matrix, x_ptr, y_ptr, end - start);
        } else if(m == 4 && n == 12) {
          templated_cuda_gemm_gpu<4,12><<<nblocks,nthread,m*sizeof(double)>>>(
                                              strideX, strideY, alpha, beta,
                                              matrix, x_ptr, y_ptr, end - start);
        } else if(m == 10 && n == 24) {
          templated_cuda_gemm_gpu<10,24><<<nblocks,nthread,m*sizeof(double)>>>(
                                              strideX, strideY, alpha, beta,
                                              matrix, x_ptr, y_ptr, end - start);
        } else if(m == 20 && n == 40) {
          templated_cuda_gemm_gpu<20,40><<<nblocks,nthread,m*sizeof(double)>>>(
                                              strideX, strideY, alpha, beta,
                                              matrix, x_ptr, y_ptr, end - start);
        } else {
          cuda_gemm_gpu<<<nblocks,nthread,m*n*sizeof(double)>>>(m, n, strideX, strideY, alpha, beta,
                                             matrix, x_ptr, y_ptr, end - start);
        }
      }
    }
  }
  op_mpi_set_dirtybit_force_halo_exchange(nargs, args, 2);
  cutilSafeCall(cudaDeviceSynchronize());
}
