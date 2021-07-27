//
// auto-generated by op2.py
//

void curl_omp4_kernel(
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

  #pragma omp target teams num_teams(num_teams) thread_limit(nthread) map(to:data0[0:dat0size],data1[0:dat1size],data2[0:dat2size],data3[0:dat3size],data4[0:dat4size],data5[0:dat5size],data6[0:dat6size],data7[0:dat7size],data8[0:dat8size])
  #pragma omp distribute parallel for schedule(static,1)
  for ( int n_op=0; n_op<count; n_op++ ){
    //variable mapping
    const double *div0 = &data0[DG_NP*n_op];
    const double *div1 = &data1[DG_NP*n_op];
    const double *div2 = &data2[DG_NP*n_op];
    const double *div3 = &data3[DG_NP*n_op];
    const double *rx = &data4[DG_NP*n_op];
    const double *sx = &data5[DG_NP*n_op];
    const double *ry = &data6[DG_NP*n_op];
    const double *sy = &data7[DG_NP*n_op];
    double *res = &data8[DG_NP*n_op];

    //inline function
    
    for(int i = 0; i < 15; i++) {
      res[i] = rx[i] * div2[i] + sx[i] * div3[i] - ry[i] * div0[i] - sy[i] * div1[i];
    }
    //end inline func
  }

}
