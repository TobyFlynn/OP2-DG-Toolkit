//
// auto-generated by op2.py
//

//user function
inline void init_cubature(double *rx, double *sx, double *ry, double *sy,
                          double *J, double *temp) {
  // J = -xs.*yr + xr.*ys
  for(int i = 0; i < DG_CUB_NP; i++) {
    J[i] = -sx[i] * ry[i] + rx[i] * sy[i];
  }

  // rx = ys./J; sx =-yr./J; ry =-xs./J; sy = xr./J;
  for(int i = 0; i < DG_CUB_NP; i++) {
    double rx_n = sy[i] / J[i];
    double sx_n = -ry[i] / J[i];
    double ry_n = -sx[i] / J[i];
    double sy_n = rx[i] / J[i];
    rx[i] = rx_n;
    sx[i] = sx_n;
    ry[i] = ry_n;
    sy[i] = sy_n;
  }

  for(int j = 0; j < DG_NP; j++) {
    for(int i = 0; i < DG_CUB_NP; i++) {
      int ind = j * DG_CUB_NP + i;
      temp[ind] = J[i] * cubW_g[i] * cubV_g[ind];
    }
  }
}

// host stub function
void op_par_loop_init_cubature(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3,
  op_arg arg4,
  op_arg arg5){

  int nargs = 6;
  op_arg args[6];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;
  args[4] = arg4;
  args[5] = arg5;
  //create aligned pointers for dats
  ALIGNED_double       double * __restrict__ ptr0 = (double *) arg0.data;
  DECLARE_PTR_ALIGNED(ptr0,double_ALIGN);
  ALIGNED_double       double * __restrict__ ptr1 = (double *) arg1.data;
  DECLARE_PTR_ALIGNED(ptr1,double_ALIGN);
  ALIGNED_double       double * __restrict__ ptr2 = (double *) arg2.data;
  DECLARE_PTR_ALIGNED(ptr2,double_ALIGN);
  ALIGNED_double       double * __restrict__ ptr3 = (double *) arg3.data;
  DECLARE_PTR_ALIGNED(ptr3,double_ALIGN);
  ALIGNED_double       double * __restrict__ ptr4 = (double *) arg4.data;
  DECLARE_PTR_ALIGNED(ptr4,double_ALIGN);
  ALIGNED_double       double * __restrict__ ptr5 = (double *) arg5.data;
  DECLARE_PTR_ALIGNED(ptr5,double_ALIGN);

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(0);
  op_timers_core(&cpu_t1, &wall_t1);


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  init_cubature");
  }

  int exec_size = op_mpi_halo_exchanges(set, nargs, args);

  if (exec_size >0) {

    #ifdef VECTORIZE
    #pragma novector
    for ( int n=0; n<(exec_size/SIMD_VEC)*SIMD_VEC; n+=SIMD_VEC ){
      #pragma omp simd simdlen(SIMD_VEC)
      for ( int i=0; i<SIMD_VEC; i++ ){
        init_cubature(
          &(ptr0)[DG_CUB_NP * (n+i)],
          &(ptr1)[DG_CUB_NP * (n+i)],
          &(ptr2)[DG_CUB_NP * (n+i)],
          &(ptr3)[DG_CUB_NP * (n+i)],
          &(ptr4)[DG_CUB_NP * (n+i)],
          &(ptr5)[DG_CUB_NP * DG_NP * (n+i)]);
      }
    }
    //remainder
    for ( int n=(exec_size/SIMD_VEC)*SIMD_VEC; n<exec_size; n++ ){
    #else
    for ( int n=0; n<exec_size; n++ ){
    #endif
      init_cubature(
        &(ptr0)[DG_CUB_NP*n],
        &(ptr1)[DG_CUB_NP*n],
        &(ptr2)[DG_CUB_NP*n],
        &(ptr3)[DG_CUB_NP*n],
        &(ptr4)[DG_CUB_NP*n],
        &(ptr5)[DG_CUB_NP * DG_NP*n]);
    }
  }

  // combine reduction data
  op_mpi_set_dirtybit(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[0].name      = name;
  OP_kernels[0].count    += 1;
  OP_kernels[0].time     += wall_t2 - wall_t1;
  OP_kernels[0].transfer += (float)set->size * arg0.size * 2.0f;
  OP_kernels[0].transfer += (float)set->size * arg1.size * 2.0f;
  OP_kernels[0].transfer += (float)set->size * arg2.size * 2.0f;
  OP_kernels[0].transfer += (float)set->size * arg3.size * 2.0f;
  OP_kernels[0].transfer += (float)set->size * arg4.size * 2.0f;
  OP_kernels[0].transfer += (float)set->size * arg5.size * 2.0f;
}
