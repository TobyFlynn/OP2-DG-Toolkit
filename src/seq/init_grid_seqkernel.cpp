//
// auto-generated by op2.py
//

//user function
#include "../kernels/init_grid.h"

// host stub function
void op_par_loop_init_grid(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3,
  op_arg arg4,
  op_arg arg5,
  op_arg arg6,
  op_arg arg7,
  op_arg arg8){

  int nargs = 9;
  op_arg args[9];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;
  args[4] = arg4;
  args[5] = arg5;
  args[6] = arg6;
  args[7] = arg7;
  args[8] = arg8;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(3);
  op_timers_core(&cpu_t1, &wall_t1);


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  init_grid");
  }

  int set_size = op_mpi_halo_exchanges(set, nargs, args);

  if (set_size > 0) {

    for ( int n=0; n<set_size; n++ ){
      init_grid(
        &((double*)arg0.data)[DG_NP*n],
        &((double*)arg1.data)[DG_NP*n],
        &((double*)arg2.data)[DG_NP*n],
        &((double*)arg3.data)[DG_NP*n],
        &((double*)arg4.data)[3 * DG_NPF*n],
        &((double*)arg5.data)[3 * DG_NPF*n],
        &((double*)arg6.data)[DG_NP*n],
        &((double*)arg7.data)[3 * DG_NPF*n],
        &((double*)arg8.data)[3 * DG_NPF*n]);
    }
  }

  // combine reduction data
  op_mpi_set_dirtybit(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[3].name      = name;
  OP_kernels[3].count    += 1;
  OP_kernels[3].time     += wall_t2 - wall_t1;
  OP_kernels[3].transfer += (float)set->size * arg0.size * 2.0f;
  OP_kernels[3].transfer += (float)set->size * arg1.size * 2.0f;
  OP_kernels[3].transfer += (float)set->size * arg2.size * 2.0f;
  OP_kernels[3].transfer += (float)set->size * arg3.size * 2.0f;
  OP_kernels[3].transfer += (float)set->size * arg4.size * 2.0f;
  OP_kernels[3].transfer += (float)set->size * arg5.size * 2.0f;
  OP_kernels[3].transfer += (float)set->size * arg6.size * 2.0f;
  OP_kernels[3].transfer += (float)set->size * arg7.size * 2.0f;
  OP_kernels[3].transfer += (float)set->size * arg8.size * 2.0f;
}
