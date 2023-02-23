#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

#include "dg_compiler_defs.h"

template<int dg_np, int dg_np_max>
__device__ void _grad_3d_gpu(const int ind, const int *p, const double *dr, const double *ds,
                             const double *dt, const double *u,
                             const double *rx, const double *sx, const double *tx,
                             const double *ry, const double *sy, const double *ty,
                             const double *rz, const double *sz, const double *tz,
                             double *ux, double *uy, double *uz) {
  if(ind > dg_np) return;

  const double *dr_mat = &dr[(*p - 1) * dg_np_max * dg_np_max];
  const double *ds_mat = &ds[(*p - 1) * dg_np_max * dg_np_max];
  const double *dt_mat = &dt[(*p - 1) * dg_np_max * dg_np_max];

  double tmp_r = 0.0;
  double tmp_s = 0.0;
  double tmp_t = 0.0;
  for(int n = 0; n < dg_np; n++) {
    int mat_ind = DG_MAT_IND(ind, n, dg_np, dg_np);
    tmp_r += dr_mat[mat_ind] * u[n];
    tmp_s += ds_mat[mat_ind] * u[n];
    tmp_t += dt_mat[mat_ind] * u[n];
  }
  ux[ind] = *rx * tmp_r + *sx * tmp_s + *tx * tmp_t;
  uy[ind] = *ry * tmp_r + *sy * tmp_s + *ty * tmp_t;
  uz[ind] = *rz * tmp_r + *sz * tmp_s + *tz * tmp_t;
}

// CUDA kernel function
template<int NUM_CELLS>
__global__ void _op_cuda_grad_3d(
  const int *__restrict arg0,
  const double *arg1,
  const double *arg2,
  const double *arg3,
  const double *__restrict arg4,
  const double *__restrict arg5,
  const double *__restrict arg6,
  const double *__restrict arg7,
  const double *__restrict arg8,
  const double *__restrict arg9,
  const double *__restrict arg10,
  const double *__restrict arg11,
  const double *__restrict arg12,
  const double *__restrict arg13,
  double *arg14,
  double *arg15,
  double *arg16,
  int   set_size ) {

  // Load matrices into shared memory
  __shared__ double dr_shared[DG_ORDER * DG_NP * DG_NP];
  __shared__ double ds_shared[DG_ORDER * DG_NP * DG_NP];
  __shared__ double dt_shared[DG_ORDER * DG_NP * DG_NP];
  __shared__ double u_shared[NUM_CELLS * DG_NP];
  __shared__ double ux_shared[NUM_CELLS * DG_NP];
  __shared__ double uy_shared[NUM_CELLS * DG_NP];
  __shared__ double uz_shared[NUM_CELLS * DG_NP];

  for(int i = threadIdx.x; i < DG_ORDER * DG_NP * DG_NP; i += blockDim.x) {
    dr_shared[i] = arg1[i];
    ds_shared[i] = arg2[i];
    dt_shared[i] = arg3[i];
  }

  __syncthreads();

  //process set elements
  for (int n = (threadIdx.x / DG_NP) + blockIdx.x * NUM_CELLS; n < set_size; n += NUM_CELLS * gridDim.x){
    // If entire thread is in set
    if(n - (threadIdx.x / DG_NP) + NUM_CELLS < set_size) {
      __syncthreads();
      u_shared[threadIdx.x] = arg4[n * DG_NP + threadIdx.x % DG_NP];
      __syncthreads();

      switch(*(arg0+n*1)) {
        case 1:
        _grad_3d_gpu<4, DG_NP>(threadIdx.x % DG_NP, arg0+n*1,
                dr_shared,
                ds_shared,
                dt_shared,
                u_shared + (threadIdx.x / DG_NP) * DG_NP, //arg4+n*DG_NP,
                arg5+n*1,
                arg6+n*1,
                arg7+n*1,
                arg8+n*1,
                arg9+n*1,
                arg10+n*1,
                arg11+n*1,
                arg12+n*1,
                arg13+n*1,
                ux_shared + (threadIdx.x / DG_NP) * DG_NP, //arg14+n*DG_NP,
                uy_shared + (threadIdx.x / DG_NP) * DG_NP, //arg15+n*DG_NP,
                uz_shared + (threadIdx.x / DG_NP) * DG_NP); //arg16+n*DG_NP);
          break;
        case 2:
        _grad_3d_gpu<10, DG_NP>(threadIdx.x % DG_NP, arg0+n*1,
                dr_shared,
                ds_shared,
                dt_shared,
                u_shared + (threadIdx.x / DG_NP) * DG_NP, //arg4+n*DG_NP,
                arg5+n*1,
                arg6+n*1,
                arg7+n*1,
                arg8+n*1,
                arg9+n*1,
                arg10+n*1,
                arg11+n*1,
                arg12+n*1,
                arg13+n*1,
                ux_shared + (threadIdx.x / DG_NP) * DG_NP, //arg14+n*DG_NP,
                uy_shared + (threadIdx.x / DG_NP) * DG_NP, //arg15+n*DG_NP,
                uz_shared + (threadIdx.x / DG_NP) * DG_NP); //arg16+n*DG_NP);
          break;
        case 3:
        _grad_3d_gpu<20, DG_NP>(threadIdx.x % DG_NP, arg0+n*1,
                dr_shared,
                ds_shared,
                dt_shared,
                u_shared + (threadIdx.x / DG_NP) * DG_NP, //arg4+n*DG_NP,
                arg5+n*1,
                arg6+n*1,
                arg7+n*1,
                arg8+n*1,
                arg9+n*1,
                arg10+n*1,
                arg11+n*1,
                arg12+n*1,
                arg13+n*1,
                ux_shared + (threadIdx.x / DG_NP) * DG_NP, //arg14+n*DG_NP,
                uy_shared + (threadIdx.x / DG_NP) * DG_NP, //arg15+n*DG_NP,
                uz_shared + (threadIdx.x / DG_NP) * DG_NP); //arg16+n*DG_NP);
          break;
      }
      __syncthreads();
      arg14[n * DG_NP + threadIdx.x % DG_NP] = ux_shared[threadIdx.x];
      arg15[n * DG_NP + threadIdx.x % DG_NP] = uy_shared[threadIdx.x];
      arg16[n * DG_NP + threadIdx.x % DG_NP] = uz_shared[threadIdx.x];
      __syncthreads();
      // __syncthreads();
      // for(int i = threadIdx.x; i < 20 * blockDim.x; i += blockDim.x) {
      //   arg14[start_ind + i] = ux_shared[i];
      //   arg15[start_ind + i] = uy_shared[i];
      //   arg16[start_ind + i] = uz_shared[i];
      // }
      // int start_ind = n * 20;
      // for(int i = 0; i < 20; i++) {
      //   u_shared[threadIdx.x * 20 + i] = arg4[start_ind + i];
      // }
    } else {
      switch(*(arg0+n*1)) {
        case 1:
        _grad_3d_gpu<4, DG_NP>(threadIdx.x % DG_NP, arg0+n*1,
                dr_shared,
                ds_shared,
                dt_shared,
                arg4+n*DG_NP,
                arg5+n*1,
                arg6+n*1,
                arg7+n*1,
                arg8+n*1,
                arg9+n*1,
                arg10+n*1,
                arg11+n*1,
                arg12+n*1,
                arg13+n*1,
                arg14+n*DG_NP,
                arg15+n*DG_NP,
                arg16+n*DG_NP);
          break;
        case 2:
        _grad_3d_gpu<10, DG_NP>(threadIdx.x % DG_NP, arg0+n*1,
                dr_shared,
                ds_shared,
                dt_shared,
                arg4+n*DG_NP,
                arg5+n*1,
                arg6+n*1,
                arg7+n*1,
                arg8+n*1,
                arg9+n*1,
                arg10+n*1,
                arg11+n*1,
                arg12+n*1,
                arg13+n*1,
                arg14+n*DG_NP,
                arg15+n*DG_NP,
                arg16+n*DG_NP);
          break;
        case 3:
        _grad_3d_gpu<20, DG_NP>(threadIdx.x % DG_NP, arg0+n*1,
                dr_shared,
                ds_shared,
                dt_shared,
                arg4+n*DG_NP,
                arg5+n*1,
                arg6+n*1,
                arg7+n*1,
                arg8+n*1,
                arg9+n*1,
                arg10+n*1,
                arg11+n*1,
                arg12+n*1,
                arg13+n*1,
                arg14+n*DG_NP,
                arg15+n*DG_NP,
                arg16+n*DG_NP);
          break;
      }
    }
  }
}


//host stub function
void custom_kernel_grad_3d(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3,
  op_arg arg4,
  op_arg arg5,
  op_arg arg6,
  op_arg arg7,
  op_arg arg8,
  op_arg arg9,
  op_arg arg10,
  op_arg arg11,
  op_arg arg12,
  op_arg arg13,
  op_arg arg14,
  op_arg arg15,
  op_arg arg16){

  double*arg1h = (double *)arg1.data;
  double*arg2h = (double *)arg2.data;
  double*arg3h = (double *)arg3.data;
  int nargs = 17;
  op_arg args[17];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;
  args[4] = arg4;
  args[5] = arg5;
  args[6] = arg6;
  args[7] = arg7;
  args[8] = arg8;
  args[9] = arg9;
  args[10] = arg10;
  args[11] = arg11;
  args[12] = arg12;
  args[13] = arg13;
  args[14] = arg14;
  args[15] = arg15;
  args[16] = arg16;


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  grad_3d");
  }

  int set_size = op_mpi_halo_exchanges_cuda(set, nargs, args);
  if (set_size > 0) {

    //transfer constants to GPU
    int consts_bytes = 0;
    consts_bytes += ROUND_UP(1200*sizeof(double));
    consts_bytes += ROUND_UP(1200*sizeof(double));
    consts_bytes += ROUND_UP(1200*sizeof(double));
    reallocConstArrays(consts_bytes);
    consts_bytes = 0;
    arg1.data   = OP_consts_h + consts_bytes;
    arg1.data_d = OP_consts_d + consts_bytes;
    for ( int d=0; d<1200; d++ ){
      ((double *)arg1.data)[d] = arg1h[d];
    }
    consts_bytes += ROUND_UP(1200*sizeof(double));
    arg2.data   = OP_consts_h + consts_bytes;
    arg2.data_d = OP_consts_d + consts_bytes;
    for ( int d=0; d<1200; d++ ){
      ((double *)arg2.data)[d] = arg2h[d];
    }
    consts_bytes += ROUND_UP(1200*sizeof(double));
    arg3.data   = OP_consts_h + consts_bytes;
    arg3.data_d = OP_consts_d + consts_bytes;
    for ( int d=0; d<1200; d++ ){
      ((double *)arg3.data)[d] = arg3h[d];
    }
    consts_bytes += ROUND_UP(1200*sizeof(double));
    mvConstArraysToDevice(consts_bytes);

    //set CUDA execution parameters
    #ifdef OP_BLOCK_SIZE_19
      int nthread = OP_BLOCK_SIZE_19;
    #else
      int nthread = OP_block_size;
    #endif
    int nblocks = 200;
    const int num_cells = 20;
    nthread = num_cells * DG_NP;

    _op_cuda_grad_3d<num_cells><<<nblocks,nthread>>>(
      (int *) arg0.data_d,
      (double *) arg1.data_d,
      (double *) arg2.data_d,
      (double *) arg3.data_d,
      (double *) arg4.data_d,
      (double *) arg5.data_d,
      (double *) arg6.data_d,
      (double *) arg7.data_d,
      (double *) arg8.data_d,
      (double *) arg9.data_d,
      (double *) arg10.data_d,
      (double *) arg11.data_d,
      (double *) arg12.data_d,
      (double *) arg13.data_d,
      (double *) arg14.data_d,
      (double *) arg15.data_d,
      (double *) arg16.data_d,
      set->size );
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
}
