//
// auto-generated by op2.py
//

//user function
__device__ void curl_gpu( const int *p, const double *div0, const double *div1,
                 const double *div2, const double *div3, const double *rx,
                 const double *sx, const double *ry, const double *sy,
                 double *res) {

  const int dg_np  = DG_CONSTANTS_cuda[(*p - 1) * 5];

  for(int i = 0; i < dg_np; i++) {
    res[i] = rx[i] * div2[i] + sx[i] * div3[i] - ry[i] * div0[i] - sy[i] * div1[i];
  }

}

// CUDA kernel function
__global__ void op_cuda_curl(
  const int *__restrict arg0,
  const double *__restrict arg1,
  const double *__restrict arg2,
  const double *__restrict arg3,
  const double *__restrict arg4,
  const double *__restrict arg5,
  const double *__restrict arg6,
  const double *__restrict arg7,
  const double *__restrict arg8,
  double *arg9,
  int   set_size ) {


  //process set elements
  for ( int n=threadIdx.x+blockIdx.x*blockDim.x; n<set_size; n+=blockDim.x*gridDim.x ){

    //user-supplied kernel call
    curl_gpu(arg0+n*1,
         arg1+n*DG_NP,
         arg2+n*DG_NP,
         arg3+n*DG_NP,
         arg4+n*DG_NP,
         arg5+n*DG_NP,
         arg6+n*DG_NP,
         arg7+n*DG_NP,
         arg8+n*DG_NP,
         arg9+n*DG_NP);
  }
}


//host stub function
void op_par_loop_curl(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3,
  op_arg arg4,
  op_arg arg5,
  op_arg arg6,
  op_arg arg7,
  op_arg arg8,
  op_arg arg9){

  int nargs = 10;
  op_arg args[10];

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

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(11);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[11].name      = name;
  OP_kernels[11].count    += 1;


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  curl");
  }

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2);
  if (set_size > 0) {

    //set CUDA execution parameters
    #ifdef OP_BLOCK_SIZE_11
      int nthread = OP_BLOCK_SIZE_11;
    #else
      int nthread = OP_block_size;
    #endif

    int nblocks = 200;

    op_cuda_curl<<<nblocks,nthread>>>(
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
      set->size );
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
  //update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[11].time     += wall_t2 - wall_t1;
  OP_kernels[11].transfer += (float)set->size * arg0.size;
  OP_kernels[11].transfer += (float)set->size * arg1.size;
  OP_kernels[11].transfer += (float)set->size * arg2.size;
  OP_kernels[11].transfer += (float)set->size * arg3.size;
  OP_kernels[11].transfer += (float)set->size * arg4.size;
  OP_kernels[11].transfer += (float)set->size * arg5.size;
  OP_kernels[11].transfer += (float)set->size * arg6.size;
  OP_kernels[11].transfer += (float)set->size * arg7.size;
  OP_kernels[11].transfer += (float)set->size * arg8.size;
  OP_kernels[11].transfer += (float)set->size * arg9.size * 2.0f;
}
