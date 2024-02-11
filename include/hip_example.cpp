// Example with aligned memory
__constant__ int direct_aligned_example;
//user function
__device__ void grad_aligned_gpu(const int *p, const double *geof,
                           const double * __restrict__ fact,
                           double * __restrict__ ux, double * __restrict__ uy,
                           double *__restrict__ uz) {
  const int dg_np = DG_CONSTANTS_TK_cuda[(*p - 1) * DG_NUM_CONSTANTS];

  const double rx = geof[(RX_IND)*direct_aligned_example];
  const double sx = geof[(SX_IND)*direct_aligned_example];
  const double tx = geof[(TX_IND)*direct_aligned_example];
  const double ry = geof[(RY_IND)*direct_aligned_example];
  const double sy = geof[(SY_IND)*direct_aligned_example];
  const double ty = geof[(TY_IND)*direct_aligned_example];
  const double rz = geof[(RZ_IND)*direct_aligned_example];
  const double sz = geof[(SZ_IND)*direct_aligned_example];
  const double tz = geof[(TZ_IND)*direct_aligned_example];
  for(int m = 0; m < dg_np; m++) {
    const double r = ux[(m)*direct_aligned_example];
    const double s = uy[(m)*direct_aligned_example];
    const double t = uz[(m)*direct_aligned_example];
    const double _fact = fact[(m)*direct_aligned_example];
    ux[(m)*direct_aligned_example] = _fact * (rx * r + sx * s + tx * t);
    uy[(m)*direct_aligned_example] = _fact * (ry * r + sy * s + ty * t);
    uz[(m)*direct_aligned_example] = _fact * (rz * r + sz * s + tz * t);
  }

}

// CUDA kernel function
__global__ void op_grad_aligned(
  const int *arg0,
  const double *__restrict arg1,
  const double *__restrict arg2,
  double *arg3,
  double *arg4,
  double *arg5,
  int start,
  int end) {


  //process set elements
  int n = threadIdx.x+blockIdx.x*blockDim.x + start;
  if (n < end) {

    //user-supplied kernel call
    grad_aligned_gpu(arg0,
                   arg1+n,
                   arg2+n,
                   arg3+n,
                   arg4+n,
                   arg5+n);
  }
}

void run_aligned_hip_example() {
  const int direct_aligned_example_HOST = 243648;
  hipMemcpyToSymbol(HIP_SYMBOL(direct_aligned_example),&direct_aligned_example_HOST,sizeof(int));

  double *ones = (double *)malloc(direct_aligned_example_HOST * DG_NP * sizeof(double));
  for(int i = 0; i < direct_aligned_example_HOST * DG_NP; i++) {
    ones[i] = 1.0;
  }

  double *ones_10 = (double *)malloc(direct_aligned_example_HOST * 10 * sizeof(double));
  for(int i = 0; i < direct_aligned_example_HOST * 10; i++) {
    ones_10[i] = 1.0;
  }

  double *fact_d, *ux_d, *uy_d, *uz_d, *geof_d;
  hipMalloc(&fact_d, direct_aligned_example_HOST * DG_NP * sizeof(double));
  hipMalloc(&ux_d, direct_aligned_example_HOST * DG_NP * sizeof(double));
  hipMalloc(&uy_d, direct_aligned_example_HOST * DG_NP * sizeof(double));
  hipMalloc(&uz_d, direct_aligned_example_HOST * DG_NP * sizeof(double));
  hipMalloc(&geof_d, direct_aligned_example_HOST * 10 * sizeof(double));

  hipMemcpy(fact_d, ones, direct_aligned_example_HOST * DG_NP * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(ux_d, ones, direct_aligned_example_HOST * DG_NP * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(uy_d, ones, direct_aligned_example_HOST * DG_NP * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(uz_d, ones, direct_aligned_example_HOST * DG_NP * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(geof_d, ones_10, direct_aligned_example_HOST * 10 * sizeof(double), hipMemcpyHostToDevice);

  int *order_d;
  int order = DG_ORDER;
  hipMalloc(&order_d, sizeof(int));
  hipMemcpy(order_d, &order, sizeof(int), hipMemcpyHostToDevice);

  const int nthread = 64;
  const int nblocks = (direct_aligned_example-1)/nthread+1;
  op_grad_aligned<<<nblocks,nthread>>>(order_d, geof_d, fact_d, ux_d, uy_d, uz_d, 0, direct_aligned_example_HOST);

  hipFree(fact_d);
  hipFree(ux_d);
  hipFree(uy_d);
  hipFree(uz_d);
  hipFree(geof_d);
  hipFree(order_d);

  free(ones);
  free(ones_10);
}