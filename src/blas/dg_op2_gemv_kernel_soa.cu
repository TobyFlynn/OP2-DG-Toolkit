#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

#include "dg_compiler_defs.h"

#include "kernels/non_templated_soa.h"
#include "kernels/templated_soa.h"

void custom_kernel_gemv(op_set set, const bool t, const int m, const int n, const DG_FP alpha,
  const DG_FP beta, const DG_FP *matrix, op_dat x, op_dat y) {

  int nargs = 2;
  op_arg args[2] = {
    op_arg_dat(x, -1, OP_ID, x->dim, DG_FP_STR, OP_READ),
    op_arg_dat(y, -1, OP_ID, y->dim, DG_FP_STR, OP_WRITE)
  };

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2);
  if (set_size > 0) {
    DG_FP *matrix_d;
    cudaMalloc(&matrix_d, m * n * sizeof(DG_FP));
    cudaMemcpy(matrix_d, matrix, m * n * sizeof(DG_FP), cudaMemcpyHostToDevice);

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
                                              matrix_d, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 10:
          templated_cuda_gemm_T_gpu<10><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                              strideX, strideY, alpha, beta,
                                              matrix_d, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 20:
          templated_cuda_gemm_T_gpu<20><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                              strideX, strideY, alpha, beta,
                                              matrix_d, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        // The number of face nodes for each order
        case 12:
          templated_cuda_gemm_T_gpu<12><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                              strideX, strideY, alpha, beta,
                                              matrix_d, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 24:
          templated_cuda_gemm_T_gpu<24><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                              strideX, strideY, alpha, beta,
                                              matrix_d, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 40:
          templated_cuda_gemm_T_gpu<40><<<nblocks,nthread,m*n*sizeof(double)>>>(n,
                                              strideX, strideY, alpha, beta,
                                              matrix_d, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        default:
          cuda_gemm_T_gpu<<<nblocks,nthread,m*n*sizeof(double)>>>(m, n, strideX, strideY, alpha, beta,
                                               matrix_d, (double *) args[0].data_d,
                                               (double *) args[1].data_d, set->size);
      }
    } else {
      switch(n) {
        // The number of nodes for each order
        case 4:
          templated_cuda_gemm_gpu<4><<<nblocks,nthread,m*n*sizeof(double)>>>(m,
                                              strideX, strideY, alpha, beta,
                                              matrix_d, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 10:
          templated_cuda_gemm_gpu<10><<<nblocks,nthread,m*n*sizeof(double)>>>(m,
                                              strideX, strideY, alpha, beta,
                                              matrix_d, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 20:
          templated_cuda_gemm_gpu<20><<<nblocks,nthread,m*n*sizeof(double)>>>(m,
                                              strideX, strideY, alpha, beta,
                                              matrix_d, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        // The number of face nodes for each order
        case 12:
          templated_cuda_gemm_gpu<12><<<nblocks,nthread,m*n*sizeof(double)>>>(m,
                                              strideX, strideY, alpha, beta,
                                              matrix_d, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 24:
          templated_cuda_gemm_gpu<24><<<nblocks,nthread,m*n*sizeof(double)>>>(m,
                                              strideX, strideY, alpha, beta,
                                              matrix_d, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        case 40:
          templated_cuda_gemm_gpu<40><<<nblocks,nthread,m*n*sizeof(double)>>>(m,
                                              strideX, strideY, alpha, beta,
                                              matrix_d, (double *) args[0].data_d,
                                              (double *) args[1].data_d, set->size);
          break;
        default:
          cuda_gemm_gpu<<<nblocks,nthread,m*n*sizeof(double)>>>(m, n, strideX, strideY, alpha, beta,
                                             matrix_d, (double *) args[0].data_d,
                                             (double *) args[1].data_d, set->size);
      }
    }

    cudaFree(matrix_d);
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
}
