#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

#include "dg_compiler_defs.h"

#include "dg_global_constants/dg_mat_constants_3d.h"
#include "dg_mesh/dg_mesh_3d.h"

__constant__ int direct_custom_grad_3d_he_stride_OP2CONSTANT;
int direct_custom_grad_3d_he_stride_OP2HOST=-1;

template<int dg_np>
__device__ void grad_3d_he_gpu(const double *geof, const double *ur,
                       const double *us, const double *ut, double *ux, double *uy,
                       double *uz) {
  for(int m = 0; m < dg_np; m++) {
    ux[(m)*direct_custom_grad_3d_he_stride_OP2CONSTANT] = geof[(RX_IND)*direct_custom_grad_3d_he_stride_OP2CONSTANT] * ur[(m)*direct_custom_grad_3d_he_stride_OP2CONSTANT] + geof[(SX_IND)*direct_custom_grad_3d_he_stride_OP2CONSTANT] * us[(m)*direct_custom_grad_3d_he_stride_OP2CONSTANT] + geof[(TX_IND)*direct_custom_grad_3d_he_stride_OP2CONSTANT] * ut[(m)*direct_custom_grad_3d_he_stride_OP2CONSTANT];
    uy[(m)*direct_custom_grad_3d_he_stride_OP2CONSTANT] = geof[(RY_IND)*direct_custom_grad_3d_he_stride_OP2CONSTANT] * ur[(m)*direct_custom_grad_3d_he_stride_OP2CONSTANT] + geof[(SY_IND)*direct_custom_grad_3d_he_stride_OP2CONSTANT] * us[(m)*direct_custom_grad_3d_he_stride_OP2CONSTANT] + geof[(TY_IND)*direct_custom_grad_3d_he_stride_OP2CONSTANT] * ut[(m)*direct_custom_grad_3d_he_stride_OP2CONSTANT];
    uz[(m)*direct_custom_grad_3d_he_stride_OP2CONSTANT] = geof[(RZ_IND)*direct_custom_grad_3d_he_stride_OP2CONSTANT] * ur[(m)*direct_custom_grad_3d_he_stride_OP2CONSTANT] + geof[(SZ_IND)*direct_custom_grad_3d_he_stride_OP2CONSTANT] * us[(m)*direct_custom_grad_3d_he_stride_OP2CONSTANT] + geof[(TZ_IND)*direct_custom_grad_3d_he_stride_OP2CONSTANT] * ut[(m)*direct_custom_grad_3d_he_stride_OP2CONSTANT];
  }

}

template<int dg_np>
__global__ void op_cuda_grad_3d_he(
  const double *__restrict arg1,
  const double *__restrict arg2,
  const double *__restrict arg3,
  const double *__restrict arg4,
  double *arg5,
  double *arg6,
  double *arg7,
  int   set_size ) {

  //process set elements
  int n = threadIdx.x+blockIdx.x*blockDim.x;
  if (n < set_size) {

    //user-supplied kernel call
    grad_3d_he_gpu<dg_np>(
               arg1+n,
               arg2+n,
               arg3+n,
               arg4+n,
               arg5+n,
               arg6+n,
               arg7+n);
  }
}

void custom_kernel_grad_3d_he(const int order, DGMesh3D *mesh,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3,
  op_arg arg4,
  op_arg arg5,
  op_arg arg6,
  op_arg arg7) {

  int nargs = 7;
  op_arg args[7];

  args[0] = arg1;
  args[1] = arg2;
  args[2] = arg3;
  args[3] = arg4;
  args[4] = arg5;
  args[5] = arg6;
  args[6] = arg7;

  std::vector<op_arg> args_vec;
  for(int i = 0; i < nargs; i++) {
    args_vec.push_back(op_arg_dat(args[i].dat, 0, mesh->face2cells, args[i].dim, DG_FP_STR, args[i].acc));
    args_vec.push_back(op_arg_dat(args[i].dat, 1, mesh->face2cells, args[i].dim, DG_FP_STR, args[i].acc));
  }

  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  grad_3d_he");
  }

  int set_size = op_mpi_halo_exchanges_grouped(mesh->faces, args_vec.size(), args_vec.data(), 2);
  op_mpi_wait_all_grouped(args_vec.size(), args_vec.data(), 2);
  if (set_size > 0) {

    if (direct_custom_grad_3d_he_stride_OP2HOST != getSetSizeFromOpArg(&arg1)) {
      direct_custom_grad_3d_he_stride_OP2HOST = getSetSizeFromOpArg(&arg1);
      cudaMemcpyToSymbol(direct_custom_grad_3d_he_stride_OP2CONSTANT,&direct_custom_grad_3d_he_stride_OP2HOST,sizeof(int));
    }
    //set CUDA execution parameters
    #ifdef OP_BLOCK_SIZE_39
      int nthread = OP_BLOCK_SIZE_39;
    #else
      int nthread = OP_block_size;
    #endif

    int nblocks = (set_size - 1) / nthread + 1;
    int complete_set_size = mesh->cells->size + mesh->cells->exec_size + mesh->cells->nonexec_size;

    switch(order) {
      case 1:
        op_cuda_grad_3d_he<4><<<nblocks,nthread>>>(
          (double *) arg1.data_d,
          (double *) arg2.data_d,
          (double *) arg3.data_d,
          (double *) arg4.data_d,
          (double *) arg5.data_d,
          (double *) arg6.data_d,
          (double *) arg7.data_d,
          complete_set_size);
        break;
      case 2:
        op_cuda_grad_3d_he<10><<<nblocks,nthread>>>(
          (double *) arg1.data_d,
          (double *) arg2.data_d,
          (double *) arg3.data_d,
          (double *) arg4.data_d,
          (double *) arg5.data_d,
          (double *) arg6.data_d,
          (double *) arg7.data_d,
          complete_set_size);
        break;
      case 3:
        op_cuda_grad_3d_he<20><<<nblocks,nthread>>>(
          (double *) arg1.data_d,
          (double *) arg2.data_d,
          (double *) arg3.data_d,
          (double *) arg4.data_d,
          (double *) arg5.data_d,
          (double *) arg6.data_d,
          (double *) arg7.data_d,
          complete_set_size);
        break;
    }
  }
  // op_mpi_set_dirtybit_cuda(nargs, args);
  arg5.dat->dirtybit = 0;
  arg5.dat->dirty_hd = 2;
  arg6.dat->dirtybit = 0;
  arg6.dat->dirty_hd = 2;
  arg7.dat->dirtybit = 0;
  arg7.dat->dirty_hd = 2;
  cutilSafeCall(cudaDeviceSynchronize());
}
