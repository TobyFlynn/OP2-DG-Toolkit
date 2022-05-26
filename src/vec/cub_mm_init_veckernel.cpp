//
// auto-generated by op2.py
//

//user function
inline void cub_mm_init(const int *p, const double *matrix, const double *tmp,
                        double *mm) {
  // Get constants for this element's order
  const int dg_np     = DG_CONSTANTS[(*p - 1) * 5];
  const int dg_cub_np = DG_CONSTANTS[(*p - 1) * 5 + 2];
  const double *cubV  = &matrix[(*p - 1) * DG_CUB_NP * DG_NP];

  for(int i = 0; i < dg_np; i++) {
    for(int j = 0; j < dg_np; j++) {
      int mmInd = i + j * dg_np;
      mm[mmInd] = 0.0;
      for(int k = 0; k < dg_cub_np; k++) {
        int aInd = i * dg_cub_np + k;
        int bInd = j * dg_cub_np + k;
        mm[mmInd] += cubV[aInd] * tmp[bInd];
      }
    }
  }
}

// host stub function
void op_par_loop_cub_mm_init(char const *name, op_set set,
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
  //create aligned pointers for dats
  ALIGNED_int const int * __restrict__ ptr0 = (int *) arg0.data;
  DECLARE_PTR_ALIGNED(ptr0,int_ALIGN);
  ALIGNED_double const double * __restrict__ ptr2 = (double *) arg2.data;
  DECLARE_PTR_ALIGNED(ptr2,double_ALIGN);
  ALIGNED_double       double * __restrict__ ptr3 = (double *) arg3.data;
  DECLARE_PTR_ALIGNED(ptr3,double_ALIGN);

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(1);
  op_timers_core(&cpu_t1, &wall_t1);


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  cub_mm_init");
  }

  int exec_size = op_mpi_halo_exchanges(set, nargs, args);

  if (exec_size >0) {

    #ifdef VECTORIZE
    #pragma novector
    for ( int n=0; n<(exec_size/SIMD_VEC)*SIMD_VEC; n+=SIMD_VEC ){
      double dat1[SIMD_VEC];
      for ( int i=0; i<SIMD_VEC; i++ ){
        dat1[i] = *((double*)arg1.data);
      }
      #pragma omp simd simdlen(SIMD_VEC)
      for ( int i=0; i<SIMD_VEC; i++ ){
        cub_mm_init(
          &(ptr0)[1 * (n+i)],
          &dat1[i],
          &(ptr2)[DG_CUB_NP * DG_NP * (n+i)],
          &(ptr3)[DG_NP * DG_NP * (n+i)]);
      }
      for ( int i=0; i<SIMD_VEC; i++ ){
      }
    }
    //remainder
    for ( int n=(exec_size/SIMD_VEC)*SIMD_VEC; n<exec_size; n++ ){
    #else
    for ( int n=0; n<exec_size; n++ ){
    #endif
      cub_mm_init(
        &(ptr0)[1*n],
        (double*)arg1.data,
        &(ptr2)[DG_CUB_NP * DG_NP*n],
        &(ptr3)[DG_NP * DG_NP*n]);
    }
  }

  // combine reduction data
  op_mpi_set_dirtybit(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[1].name      = name;
  OP_kernels[1].count    += 1;
  OP_kernels[1].time     += wall_t2 - wall_t1;
  OP_kernels[1].transfer += (float)set->size * arg0.size;
  OP_kernels[1].transfer += (float)set->size * arg2.size;
  OP_kernels[1].transfer += (float)set->size * arg3.size * 2.0f;
}