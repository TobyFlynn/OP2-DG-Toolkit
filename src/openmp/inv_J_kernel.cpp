//
// auto-generated by op2.py
//

//user function
#include "../kernels/inv_J.h"

// host stub function
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
  op_timing_realloc(16);
  OP_kernels[16].name      = name;
  OP_kernels[16].count    += 1;
  op_timers_core(&cpu_t1, &wall_t1);


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  inv_J");
  }

  int set_size = op_mpi_halo_exchanges(set, nargs, args);
  // set number of threads
  #ifdef _OPENMP
    int nthreads = omp_get_max_threads();
  #else
    int nthreads = 1;
  #endif

  if (set_size >0) {

    // execute plan
    #pragma omp parallel for
    for ( int thr=0; thr<nthreads; thr++ ){
      int start  = (set->size* thr)/nthreads;
      int finish = (set->size*(thr+1))/nthreads;
      for ( int n=start; n<finish; n++ ){
        inv_J(
          &((int*)arg0.data)[1*n],
          &((double*)arg1.data)[DG_NP*n],
          &((double*)arg2.data)[DG_NP*n],
          &((double*)arg3.data)[DG_NP*n]);
      }
    }
  }

  // combine reduction data
  op_mpi_set_dirtybit(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[16].time     += wall_t2 - wall_t1;
  OP_kernels[16].transfer += (float)set->size * arg0.size;
  OP_kernels[16].transfer += (float)set->size * arg1.size;
  OP_kernels[16].transfer += (float)set->size * arg2.size;
  OP_kernels[16].transfer += (float)set->size * arg3.size * 2.0f;
}
