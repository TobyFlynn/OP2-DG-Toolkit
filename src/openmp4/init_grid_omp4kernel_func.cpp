//
// auto-generated by op2.py
//

void init_grid_omp4_kernel(
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
    map(to: FMASK_ompkernel[:2])
  #pragma omp distribute parallel for schedule(static,1)
  for ( int n_op=0; n_op<count; n_op++ ){
    //variable mapping
    double *rx = &data0[DG_NP*n_op];
    double *ry = &data1[DG_NP*n_op];
    double *sx = &data2[DG_NP*n_op];
    double *sy = &data3[DG_NP*n_op];
    double *nx = &data4[DG_NP*n_op];
    double *ny = &data5[DG_NP*n_op];
    double *J = &data6[DG_NP*n_op];
    double *sJ = &data7[DG_NP*n_op];
    double *fscale = &data8[DG_NP*n_op];

    //inline function
    


    for(int i = 0; i < 5; i++) {
      nx[i] = ry[FMASK_ompkernel[i]];
      ny[i] = -rx[FMASK_ompkernel[i]];
    }

    for(int i = 0; i < 5; i++) {
      nx[5 + i] = sy[FMASK_ompkernel[5 + i]] - ry[FMASK_ompkernel[5 + i]];
      ny[5 + i] = rx[FMASK_ompkernel[5 + i]] - sx[FMASK_ompkernel[5 + i]];
    }

    for(int i = 0; i < 5; i++) {
      nx[2 * 5 + i] = -sy[FMASK_ompkernel[2 * 5 + i]];
      ny[2 * 5 + i] = sx[FMASK_ompkernel[2 * 5 + i]];
    }

    for(int i = 0; i < 15; i++) {
      J[i] = -sx[i] * ry[i] + rx[i] * sy[i];
    }

    for(int i = 0; i < 15; i++) {
      double rx_n = sy[i] / J[i];
      double sx_n = -ry[i] / J[i];
      double ry_n = -sx[i] / J[i];
      double sy_n = rx[i] / J[i];
      rx[i] = rx_n;
      sx[i] = sx_n;
      ry[i] = ry_n;
      sy[i] = sy_n;
    }

    for(int i = 0; i < 3 * 5; i++) {
      sJ[i] = sqrt(nx[i] * nx[i] + ny[i] * ny[i]);
      nx[i] = nx[i] / sJ[i];
      ny[i] = ny[i] / sJ[i];
      fscale[i] = sJ[i] / J[FMASK_ompkernel[i]];
    }
    //end inline func
  }

}
