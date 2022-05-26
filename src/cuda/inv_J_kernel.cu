//
// auto-generated by op2.py
//

//user function
__device__ void inv_J_gpu( const int *p, const double *J, const double *tmp, double *u) {

  const int dg_np = DG_CONSTANTS_cuda[(*p - 1) * 5];
  for(int i = 0; i < dg_np; i++) {
    u[i] = tmp[i] / J[i];
  }

}

// CUDA kernel function
__global__ void op_cuda_inv_J(
  const int *__restrict arg0,
  const double *__restrict arg1,
  const double *__restrict arg2,
  double *arg3,
  int   set_size ) {


  //process set elements
  for ( int n=threadIdx.x+blockIdx.x*blockDim.x; n<set_size; n+=blockDim.x*gridDim.x ){

    //user-supplied kernel call
    inv_J_gpu(arg0+n*1,
          arg1+n*DG_NP,
          arg2+n*DG_NP,
          arg3+n*DG_NP);
  }
}


//host stub function
void op_par_loop_inv_J(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3){

  int nargs = 4;
  op_arg args[4];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(28);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[28].name      = name;
  OP_kernels[28].count    += 1;


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  inv_J");
  }

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2);
  if (set_size > 0) {

    //set CUDA execution parameters
    #ifdef OP_BLOCK_SIZE_28
      int nthread = OP_BLOCK_SIZE_28;
    #else
      int nthread = OP_block_size;
    #endif

    int nblocks = 200;

    op_cuda_inv_J<<<nblocks,nthread>>>(
      (int *) arg0.data_d,
      (double *) arg1.data_d,
      (double *) arg2.data_d,
      (double *) arg3.data_d,
      set->size );
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
  //update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[28].time     += wall_t2 - wall_t1;
  OP_kernels[28].transfer += (float)set->size * arg0.size;
  OP_kernels[28].transfer += (float)set->size * arg1.size;
  OP_kernels[28].transfer += (float)set->size * arg2.size;
  OP_kernels[28].transfer += (float)set->size * arg3.size * 2.0f;
}
