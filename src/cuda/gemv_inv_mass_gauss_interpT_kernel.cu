//
// auto-generated by op2.py
//

//user function
__device__ void gemv_inv_mass_gauss_interpT_gpu( const int *p, const double *alpha,
                                        const double *beta, const double *matrix,
                                        const double *J, const double *x, double *y) {

  const int dg_np   = DG_CONSTANTS_cuda[(*p - 1) * 5];
  const int dg_g_np = DG_CONSTANTS_cuda[(*p - 1) * 5 + 3];
  const double *inv_mass_gauss_interp = &matrix[(*p - 1) * DG_G_NP * DG_NP];

  for(int i = 0; i < dg_np; i++) {
    double tmp = 0.0;
    for(int j = 0; j < dg_g_np; j++) {
      int ind = i + j * dg_np;
      tmp += *alpha * inv_mass_gauss_interp[ind] * x[j];
    }
    y[i] = *beta * y[i] + tmp / J[i];
  }

}

// CUDA kernel function
__global__ void op_cuda_gemv_inv_mass_gauss_interpT(
  const int *__restrict arg0,
  const double *arg1,
  const double *arg2,
  const double *arg3,
  const double *__restrict arg4,
  const double *__restrict arg5,
  double *arg6,
  int   set_size ) {


  //process set elements
  for ( int n=threadIdx.x+blockIdx.x*blockDim.x; n<set_size; n+=blockDim.x*gridDim.x ){

    //user-supplied kernel call
    gemv_inv_mass_gauss_interpT_gpu(arg0+n*1,
                                arg1,
                                arg2,
                                arg3,
                                arg4+n*DG_NP,
                                arg5+n*DG_G_NP,
                                arg6+n*DG_NP);
  }
}


//host stub function
void op_par_loop_gemv_inv_mass_gauss_interpT(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3,
  op_arg arg4,
  op_arg arg5,
  op_arg arg6){

  double*arg1h = (double *)arg1.data;
  double*arg2h = (double *)arg2.data;
  double*arg3h = (double *)arg3.data;
  int nargs = 7;
  op_arg args[7];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;
  args[4] = arg4;
  args[5] = arg5;
  args[6] = arg6;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(18);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[18].name      = name;
  OP_kernels[18].count    += 1;


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  gemv_inv_mass_gauss_interpT");
  }

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2);
  if (set_size > 0) {

    //transfer constants to GPU
    int consts_bytes = 0;
    consts_bytes += ROUND_UP(1*sizeof(double));
    consts_bytes += ROUND_UP(1*sizeof(double));
    consts_bytes += ROUND_UP(DG_ORDER * DG_G_NP * DG_NP*sizeof(double));
    reallocConstArrays(consts_bytes);
    consts_bytes = 0;
    arg1.data   = OP_consts_h + consts_bytes;
    arg1.data_d = OP_consts_d + consts_bytes;
    for ( int d=0; d<1; d++ ){
      ((double *)arg1.data)[d] = arg1h[d];
    }
    consts_bytes += ROUND_UP(1*sizeof(double));
    arg2.data   = OP_consts_h + consts_bytes;
    arg2.data_d = OP_consts_d + consts_bytes;
    for ( int d=0; d<1; d++ ){
      ((double *)arg2.data)[d] = arg2h[d];
    }
    consts_bytes += ROUND_UP(1*sizeof(double));
    arg3.data   = OP_consts_h + consts_bytes;
    arg3.data_d = OP_consts_d + consts_bytes;
    for ( int d=0; d<DG_ORDER * DG_G_NP * DG_NP; d++ ){
      ((double *)arg3.data)[d] = arg3h[d];
    }
    consts_bytes += ROUND_UP(DG_ORDER * DG_G_NP * DG_NP*sizeof(double));
    mvConstArraysToDevice(consts_bytes);

    //set CUDA execution parameters
    #ifdef OP_BLOCK_SIZE_18
      int nthread = OP_BLOCK_SIZE_18;
    #else
      int nthread = OP_block_size;
    #endif

    int nblocks = 200;

    op_cuda_gemv_inv_mass_gauss_interpT<<<nblocks,nthread>>>(
      (int *) arg0.data_d,
      (double *) arg1.data_d,
      (double *) arg2.data_d,
      (double *) arg3.data_d,
      (double *) arg4.data_d,
      (double *) arg5.data_d,
      (double *) arg6.data_d,
      set->size );
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
  //update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[18].time     += wall_t2 - wall_t1;
  OP_kernels[18].transfer += (float)set->size * arg0.size;
  OP_kernels[18].transfer += (float)set->size * arg4.size;
  OP_kernels[18].transfer += (float)set->size * arg5.size;
  OP_kernels[18].transfer += (float)set->size * arg6.size * 2.0f;
}
