#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

#include "dg_compiler_defs.h"

template<int dg_np>
__device__ void _mass_gpu(const int ind, const int *p, const DG_FP *matrix, const DG_FP *J, const DG_FP *in, DG_FP *x) {
  const DG_FP *mat = &matrix[(*p - 1) * dg_np * dg_np];

  DG_FP tmp = 0.0;
  for(int j = 0; j < dg_np; j++) {
    int mat_ind = DG_MAT_IND(ind, j, dg_np, dg_np);
    tmp += mat[mat_ind] * in[j];
  }
  x[ind] = *J * tmp;
}

// CUDA kernel function
template<int NUM_CELLS>
__global__ void _op_cuda_mass(
  const int *__restrict arg0,
  const DG_FP *arg1,
  const DG_FP *__restrict arg2,
  DG_FP *arg3,
  int   set_size ) {

  // Load matrices into shared memory
  __shared__ DG_FP mass_shared[DG_ORDER * DG_NP * DG_NP];
  __shared__ DG_FP u_shared[NUM_CELLS * DG_NP];

  for(int i = threadIdx.x; i < DG_ORDER * DG_NP * DG_NP; i += blockDim.x) {
    mass_shared[i] = arg1[i];
  }

  __syncthreads();

  for (int n = threadIdx.x + blockIdx.x * blockDim.x; n - threadIdx.x < set_size * DG_NP; n += blockDim.x * gridDim.x){
    const int node_id = n % DG_NP;
    const int cell_id = n / DG_NP;
    const int local_cell_id = (n / DG_NP) - ((n - threadIdx.x) / DG_NP);
    __syncthreads();
    const int start_ind = ((n - threadIdx.x) / DG_NP) * DG_NP;
    const int num_elem  = (min(n - threadIdx.x + blockDim.x, set_size * DG_NP) / DG_NP) - ((n - threadIdx.x) / DG_NP) + 1;
    for(int i = threadIdx.x; i < num_elem * DG_NP; i += blockDim.x) {
      u_shared[i] = arg3[start_ind + i];
    }
    __syncthreads();

    if(n < set_size * DG_NP) {
      switch (*(arg0 + cell_id * 1)) {
        case 1:
          _mass_gpu<4>(node_id, arg0 + cell_id * 1,
               arg1,
               arg2 + cell_id * 1,
               u_shared + local_cell_id * DG_NP,
               arg3 + cell_id * DG_NP);
          break;
        case 2:
          _mass_gpu<10>(node_id, arg0 + cell_id * 1,
               arg1,
               arg2 + cell_id * 1,
               u_shared + local_cell_id * DG_NP,
               arg3 + cell_id * DG_NP);
          break;
        case 3:
          _mass_gpu<20>(node_id, arg0 + cell_id * 1,
               arg1,
               arg2 + cell_id * 1,
               u_shared + local_cell_id * DG_NP,
               arg3 + cell_id * DG_NP);
          break;
      }
    }
  }
}


//host stub function
void custom_kernel_mass(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3){

  DG_FP*arg1h = (DG_FP *)arg1.data;
  int nargs = 4;
  op_arg args[4];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;

  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  mass");
  }

  int set_size = op_mpi_halo_exchanges_cuda(set, nargs, args);
  if (set_size > 0) {

    //transfer constants to GPU
    int consts_bytes = 0;
    consts_bytes += ROUND_UP(DG_ORDER * DG_NP * DG_NP * sizeof(DG_FP));
    reallocConstArrays(consts_bytes);
    consts_bytes = 0;
    arg1.data   = OP_consts_h + consts_bytes;
    arg1.data_d = OP_consts_d + consts_bytes;
    memcpy(arg1.data, arg1h, DG_ORDER * DG_NP * DG_NP * sizeof(DG_FP));
    // for ( int d=0; d<1200; d++ ){
    //   ((DG_FP *)arg1.data)[d] = arg1h[d];
    // }
    consts_bytes += ROUND_UP(DG_ORDER * DG_NP * DG_NP * sizeof(DG_FP));
    mvConstArraysToDevice(consts_bytes);

    //set CUDA execution parameters
    const int nthread = (256 / DG_NP) * DG_NP;
    const int nblocks = 200 < (set->size * DG_NP) / nthread + 1 ? 200 : (set->size * DG_NP) / nthread + 1;
    const int num_cells = (nthread / DG_NP) + 1;

    _op_cuda_mass<num_cells><<<nblocks,nthread>>>(
      (int *) arg0.data_d,
      (DG_FP *) arg1.data_d,
      (DG_FP *) arg2.data_d,
      (DG_FP *) arg3.data_d,
      set->size );
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
}
