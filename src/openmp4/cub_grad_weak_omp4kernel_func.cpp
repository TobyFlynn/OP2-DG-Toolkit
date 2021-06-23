//
// auto-generated by op2.py
//

void cub_grad_weak_omp4_kernel(
  double *data0,
  int dat0size,
  double *data1,
  int dat1size,
  double *data2,
  int dat2size,
  double *data3,
  int dat3size,
  double *data4,
  int dat4size,
  double *data5,
  int dat5size,
  double *data6,
  int dat6size,
  double *data7,
  int dat7size,
  double *data8,
  int dat8size,
  int count,
  int num_teams,
  int nthread){

  #pragma omp target teams num_teams(num_teams) thread_limit(nthread) map(to:data0[0:dat0size],data1[0:dat1size],data2[0:dat2size],data3[0:dat3size],data4[0:dat4size],data5[0:dat5size],data6[0:dat6size],data7[0:dat7size],data8[0:dat8size]) \
    map(to: cubW_g_ompkernel[:46])
  #pragma omp distribute parallel for schedule(static,1)
  for ( int n_op=0; n_op<count; n_op++ ){
    //variable mapping
    double *temp0 = &data0[46*n_op];
    const double *rx = &data1[46*n_op];
    const double *sx = &data2[46*n_op];
    const double *ry = &data3[46*n_op];
    const double *sy = &data4[46*n_op];
    const double *J = &data5[46*n_op];
    double *temp1 = &data6[46*n_op];
    double *temp2 = &data7[46*n_op];
    double *temp3 = &data8[46*n_op];

    //inline function
    
    for(int i = 0; i < 46; i++) {
      temp1[i] = cubW_g_ompkernel[i] * J[i] * sx[i] * temp0[i];
      temp2[i] = cubW_g_ompkernel[i] * J[i] * ry[i] * temp0[i];
      temp3[i] = cubW_g_ompkernel[i] * J[i] * sy[i] * temp0[i];
      temp0[i] = cubW_g_ompkernel[i] * J[i] * rx[i] * temp0[i];
    }
    //end inline func
  }

}
