//
// auto-generated by op2.py
//

//user function
__device__ void interp_dat_to_max_order_gpu( const double *mats, const int *order,
                                    const double *in, double *out) {

  const int dg_np_old = DG_CONSTANTS_cuda[(*order - 1) * 5];
  const int dg_np_new = DG_CONSTANTS_cuda[(DG_ORDER - 1) * 5];
  const double *mat = &mats[((*order - 1) * DG_ORDER + (DG_ORDER - 1)) * DG_NP * DG_NP];

  if(*order == DG_ORDER) {
    for(int i = 0; i < dg_np_new; i++) {
      out[i] = in[i];
    }
    return;
  }

  for(int i = 0; i < dg_np_new; i++) {
    out[i] = 0.0;
    for(int j = 0; j < dg_np_old; j++) {
      int ind = i + j * dg_np_new;
      out[i] += mat[ind] * in[j];
    }
  }

}

// CUDA kernel function
__global__ void op_cuda_interp_dat_to_max_order(
  const double *arg0,
  const int *__restrict arg1,
  const double *__restrict arg2,
  double *arg3,
  int   set_size ) {


  //process set elements
  for ( int n=threadIdx.x+blockIdx.x*blockDim.x; n<set_size; n+=blockDim.x*gridDim.x ){

    //user-supplied kernel call
    interp_dat_to_max_order_gpu(arg0,
                            arg1+n*1,
                            arg2+n*DG_NP,
                            arg3+n*DG_NP);
  }
}


//host stub function
void op_par_loop_interp_dat_to_max_order(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3){

  double*arg0h = (double *)arg0.data;
  int nargs = 4;
  op_arg args[4];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(9);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[9].name      = name;
  OP_kernels[9].count    += 1;


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  interp_dat_to_max_order");
  }

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2);
  if (set_size > 0) {

    //transfer constants to GPU
    int consts_bytes = 0;
    consts_bytes += ROUND_UP(DG_ORDER * DG_ORDER * DG_NP * DG_NP*sizeof(double));
    reallocConstArrays(consts_bytes);
    consts_bytes = 0;
    arg0.data   = OP_consts_h + consts_bytes;
    arg0.data_d = OP_consts_d + consts_bytes;
    for ( int d=0; d<DG_ORDER * DG_ORDER * DG_NP * DG_NP; d++ ){
      ((double *)arg0.data)[d] = arg0h[d];
    }
    consts_bytes += ROUND_UP(DG_ORDER * DG_ORDER * DG_NP * DG_NP*sizeof(double));
    mvConstArraysToDevice(consts_bytes);

    //set CUDA execution parameters
    #ifdef OP_BLOCK_SIZE_9
      int nthread = OP_BLOCK_SIZE_9;
    #else
      int nthread = OP_block_size;
    #endif

    int nblocks = 200;

    op_cuda_interp_dat_to_max_order<<<nblocks,nthread>>>(
      (double *) arg0.data_d,
      (int *) arg1.data_d,
      (double *) arg2.data_d,
      (double *) arg3.data_d,
      set->size );
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
  //update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[9].time     += wall_t2 - wall_t1;
  OP_kernels[9].transfer += (float)set->size * arg1.size;
  OP_kernels[9].transfer += (float)set->size * arg2.size;
  OP_kernels[9].transfer += (float)set->size * arg3.size * 2.0f;
}
