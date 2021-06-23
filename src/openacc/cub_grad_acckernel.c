//
// auto-generated by op2.py
//

//user function
//user function
//#pragma acc routine
inline void cub_grad_openacc( const double *rx, const double *sx, const double *ry,
                     const double *sy, const double *J, double *temp0,
                     double *temp1) {
  for(int i = 0; i < 46; i++) {
    double dru = temp0[i];
    double dsu = temp1[i];
    temp0[i] = cubW_g[i] * J[i] * (rx[i] * dru + sx[i] * dsu);
    temp1[i] = cubW_g[i] * J[i] * (ry[i] * dru + sy[i] * dsu);
  }
}

// host stub function
void op_par_loop_cub_grad(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3,
  op_arg arg4,
  op_arg arg5,
  op_arg arg6){

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
  op_timing_realloc(8);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[8].name      = name;
  OP_kernels[8].count    += 1;


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  cub_grad");
  }

  int set_size = op_mpi_halo_exchanges_cuda(set, nargs, args);


  if (set_size >0) {


    //Set up typed device pointers for OpenACC

    double* data0 = (double*)arg0.data_d;
    double* data1 = (double*)arg1.data_d;
    double* data2 = (double*)arg2.data_d;
    double* data3 = (double*)arg3.data_d;
    double* data4 = (double*)arg4.data_d;
    double* data5 = (double*)arg5.data_d;
    double* data6 = (double*)arg6.data_d;
    #pragma acc parallel loop independent deviceptr(data0,data1,data2,data3,data4,data5,data6)
    for ( int n=0; n<set->size; n++ ){
      cub_grad_openacc(
        &data0[46*n],
        &data1[46*n],
        &data2[46*n],
        &data3[46*n],
        &data4[46*n],
        &data5[46*n],
        &data6[46*n]);
    }
  }

  // combine reduction data
  op_mpi_set_dirtybit_cuda(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[8].time     += wall_t2 - wall_t1;
  OP_kernels[8].transfer += (float)set->size * arg0.size;
  OP_kernels[8].transfer += (float)set->size * arg1.size;
  OP_kernels[8].transfer += (float)set->size * arg2.size;
  OP_kernels[8].transfer += (float)set->size * arg3.size;
  OP_kernels[8].transfer += (float)set->size * arg4.size;
  OP_kernels[8].transfer += (float)set->size * arg5.size * 2.0f;
  OP_kernels[8].transfer += (float)set->size * arg6.size * 2.0f;
}
