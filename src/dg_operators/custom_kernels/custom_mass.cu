#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

#include "dg_compiler_defs.h"

#include "dg_global_constants/dg_mat_constants_3d.h"

template<int p, int dg_np>
__device__ void _mass_gpu(const int ind, const DG_FP *matrix, const DG_FP *J, const DG_FP *in, DG_FP *x) {
  if(!(ind < dg_np))
    return;
  DG_FP tmp = 0.0;
  for(int j = 0; j < dg_np; j++) {
    int mat_ind = DG_MAT_IND(ind, j, dg_np, dg_np);
    tmp += matrix[mat_ind] * in[j];
  }
  x[ind] = *J * tmp;
}

// CUDA kernel function
template<int p, int NUM_CELLS>
__global__ void _op_cuda_mass(
  const int *__restrict arg0,
  const DG_FP *__restrict arg2,
  DG_FP *arg3,
  int   set_size ) {
  const int np = (p + 1) * (p + 2) * (p + 3) / 6;
  // Load matrices into shared memory
  __shared__ DG_FP mass_shared[DG_NP * DG_NP];
  __shared__ DG_FP u_shared[NUM_CELLS * DG_NP];

  const int start_ind_mat = (p - 1) * DG_NP * DG_NP;
  for(int i = threadIdx.x; i < DG_NP * DG_NP; i += blockDim.x) {
    mass_shared[i] = dg_Mass_kernel[start_ind_mat + i];
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
      _mass_gpu<p,np>(node_id,
               mass_shared,
               arg2 + cell_id * 1,
               u_shared + local_cell_id * DG_NP,
               arg3 + cell_id * DG_NP);
    }
  }
}


//host stub function
void custom_kernel_mass(const int order, char const *name, op_set set,
  op_arg arg0,
  op_arg arg2,
  op_arg arg3){

  int nargs = 3;
  op_arg args[3];

  args[0] = arg0;
  args[1] = arg2;
  args[2] = arg3;

  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  mass");
  }

  int set_size = op_mpi_halo_exchanges_cuda(set, nargs, args);
  if (set_size > 0) {
    //set CUDA execution parameters
    const int nthread = (256 / DG_NP) * DG_NP;
    const int nblocks = 200 < (set->size * DG_NP) / nthread + 1 ? 200 : (set->size * DG_NP) / nthread + 1;
    const int num_cells = (nthread / DG_NP) + 1;

    switch(order) {
      case 1:
        _op_cuda_mass<1,num_cells><<<nblocks,nthread>>>(
          (int *) arg0.data_d,
          (DG_FP *) arg2.data_d,
          (DG_FP *) arg3.data_d,
          set->size );
        break;
      case 2:
        _op_cuda_mass<2,num_cells><<<nblocks,nthread>>>(
          (int *) arg0.data_d,
          (DG_FP *) arg2.data_d,
          (DG_FP *) arg3.data_d,
          set->size );
        break;
      case 3:
        _op_cuda_mass<3,num_cells><<<nblocks,nthread>>>(
          (int *) arg0.data_d,
          (DG_FP *) arg2.data_d,
          (DG_FP *) arg3.data_d,
          set->size );
        break;
      case 4:
        _op_cuda_mass<4,num_cells><<<nblocks,nthread>>>(
          (int *) arg0.data_d,
          (DG_FP *) arg2.data_d,
          (DG_FP *) arg3.data_d,
          set->size );
        break;
      case 5:
        _op_cuda_mass<5,num_cells><<<nblocks,nthread>>>(
          (int *) arg0.data_d,
          (DG_FP *) arg2.data_d,
          (DG_FP *) arg3.data_d,
          set->size );
        break;
    }
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
}
