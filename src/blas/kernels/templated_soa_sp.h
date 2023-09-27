#pragma once

#ifdef __ldcg_wrapper
#undef __ldcg_wrapper
#endif

#ifdef __stcg_wrapper
#undef __stcg_wrapper
#endif

#ifdef OP2_DG_CUDA
#define __ldcg_wrapper __ldcg
#define __stcg_wrapper(X, Y) __stcg((X), (Y))
#else
#define __ldcg_wrapper *
#define __stcg_wrapper(X, Y) *(X) = (Y)
#endif

template<int m, int n>
__global__ void templated_cuda_gemm_gpu_sp(
  const int strideX, const int strideY,
  const float alpha, const float beta,
  const float * __restrict__ matrix,
  const float * __restrict__ arg4, float * __restrict__ arg5,
  const int set_size) {

  extern __shared__ float sh_mem_sp[];

  float *mat_sh = sh_mem_sp;

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = __ldg(matrix + i);
  }
  __syncthreads();

  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if(cell < set_size) {
    const float *x = arg4 + cell;
    float *y = arg5 + cell;

    float y_tmp[m];
    float x_tmp = __ldcg_wrapper(x);
    #pragma unroll
    for(int i = 0; i < m; i++) {
      const int ind = DG_MAT_IND(i,0,m,n);
      y_tmp[i] = mat_sh[ind] * x_tmp;
    }

    #pragma unroll
    for(int j = 1; j < n - 1; j++) {
      x_tmp = __ldcg_wrapper(x + j * strideX);
      #pragma unroll
      for(int i = 0; i < m; i++) {
        const int ind = DG_MAT_IND(i,j,m,n);
        y_tmp[i] += mat_sh[ind] * x_tmp;
      }
    }

    x_tmp = __ldcg_wrapper(x + (n-1) * strideX);
    #pragma unroll
    for(int i = 0; i < m; i++) {
      const int ind = DG_MAT_IND(i,n-1,m,n);
      y_tmp[i] += mat_sh[ind] * x_tmp;
      y_tmp[i] = alpha * y_tmp[i] + beta * __ldcg_wrapper(y + i * strideY);
      __stcg_wrapper(y + i * strideY, y_tmp[i]);
    }
  }
}

template<int m, int n>
__global__ void templated_cuda_gemm_gpu_1_alpha_sp(
  const int strideX, const int strideY,
  const float beta,
  const float * __restrict__ matrix,
  const float * __restrict__ arg4, float * __restrict__ arg5,
  const int set_size) {

  extern __shared__ float sh_mem_sp[];

  float *mat_sh = sh_mem_sp;

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = __ldg(matrix + i);
  }
  __syncthreads();

  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if(cell < set_size) {
    const float *x = arg4 + cell;
    float *y = arg5 + cell;

    float y_tmp[m];
    float x_tmp = __ldcg_wrapper(x);
    #pragma unroll
    for(int i = 0; i < m; i++) {
      const int ind = DG_MAT_IND(i,0,m,n);
      y_tmp[i] = mat_sh[ind] * x_tmp;
    }

    #pragma unroll
    for(int j = 1; j < n - 1; j++) {
      x_tmp = __ldcg_wrapper(x + j * strideX);
      #pragma unroll
      for(int i = 0; i < m; i++) {
        const int ind = DG_MAT_IND(i,j,m,n);
        y_tmp[i] += mat_sh[ind] * x_tmp;
      }
    }

    x_tmp = __ldcg_wrapper(x + (n-1) * strideX);
    #pragma unroll
    for(int i = 0; i < m; i++) {
      const int ind = DG_MAT_IND(i,n-1,m,n);
      y_tmp[i] += mat_sh[ind] * x_tmp;
      y_tmp[i] = y_tmp[i] + beta * __ldcg_wrapper(y + i * strideY);
      __stcg_wrapper(y + i * strideY, y_tmp[i]);
    }
  }
}

template<int m, int n>
__global__ void templated_cuda_gemm_gpu_1_alpha_0_beta_sp(
  const int strideX, const int strideY,
  const float * __restrict__ matrix,
  const float * __restrict__ arg4, float * __restrict__ arg5,
  const int set_size) {

  extern __shared__ float sh_mem_sp[];

  float *mat_sh = sh_mem_sp;

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = __ldg(matrix + i);
  }
  __syncthreads();

  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if(cell < set_size) {
    const float *x = arg4 + cell;
    float *y = arg5 + cell;

    float y_tmp[m];
    float x_tmp = __ldcg_wrapper(x);
    #pragma unroll
    for(int i = 0; i < m; i++) {
      const int ind = DG_MAT_IND(i,0,m,n);
      y_tmp[i] = mat_sh[ind] * x_tmp;
    }

    #pragma unroll
    for(int j = 1; j < n - 1; j++) {
      x_tmp = __ldcg_wrapper(x + j * strideX);
      #pragma unroll
      for(int i = 0; i < m; i++) {
        const int ind = DG_MAT_IND(i,j,m,n);
        y_tmp[i] += mat_sh[ind] * x_tmp;
      }
    }

    x_tmp = __ldcg_wrapper(x + (n-1) * strideX);
    #pragma unroll
    for(int i = 0; i < m; i++) {
      const int ind = DG_MAT_IND(i,n-1,m,n);
      y_tmp[i] += mat_sh[ind] * x_tmp;
      __stcg_wrapper(y + i * strideY, y_tmp[i]);
    }
  }
}

template<int m, int n>
__global__ void templated_cuda_gemm_gpu_1_alpha_1_beta_sp(
  const int strideX, const int strideY,
  const float * __restrict__ matrix,
  const float * __restrict__ arg4, float * __restrict__ arg5,
  const int set_size) {

  extern __shared__ float sh_mem_sp[];

  float *mat_sh = sh_mem_sp;

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = __ldg(matrix + i);
  }
  __syncthreads();

  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if(cell < set_size) {
    const float *x = arg4 + cell;
    float *y = arg5 + cell;

    float y_tmp[m];
    float x_tmp = __ldcg_wrapper(x);
    #pragma unroll
    for(int i = 0; i < m; i++) {
      const int ind = DG_MAT_IND(i,0,m,n);
      y_tmp[i] = mat_sh[ind] * x_tmp;
    }

    #pragma unroll
    for(int j = 1; j < n - 1; j++) {
      x_tmp = __ldcg_wrapper(x + j * strideX);
      #pragma unroll
      for(int i = 0; i < m; i++) {
        const int ind = DG_MAT_IND(i,j,m,n);
        y_tmp[i] += mat_sh[ind] * x_tmp;
      }
    }

    x_tmp = __ldcg_wrapper(x + (n-1) * strideX);
    #pragma unroll
    for(int i = 0; i < m; i++) {
      const int ind = DG_MAT_IND(i,n-1,m,n);
      y_tmp[i] += mat_sh[ind] * x_tmp;
      y_tmp[i] = y_tmp[i] + __ldcg_wrapper(y + i * strideY);
      __stcg_wrapper(y + i * strideY, y_tmp[i]);
    }
  }
}

template<int m, int n>
__global__ void templated_cuda_gemm_gpu_0_beta_sp(
  const int strideX, const int strideY,
  const float alpha,
  const float * __restrict__ matrix,
  const float * __restrict__ arg4, float * __restrict__ arg5,
  const int set_size) {

  extern __shared__ float sh_mem_sp[];

  float *mat_sh = sh_mem_sp;

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = __ldg(matrix + i);
  }
  __syncthreads();

  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if(cell < set_size) {
    const float *x = arg4 + cell;
    float *y = arg5 + cell;

    float y_tmp[m];
    float x_tmp = __ldcg_wrapper(x);
    #pragma unroll
    for(int i = 0; i < m; i++) {
      const int ind = DG_MAT_IND(i,0,m,n);
      y_tmp[i] = mat_sh[ind] * x_tmp;
    }

    #pragma unroll
    for(int j = 1; j < n - 1; j++) {
      x_tmp = __ldcg_wrapper(x + j * strideX);
      #pragma unroll
      for(int i = 0; i < m; i++) {
        const int ind = DG_MAT_IND(i,j,m,n);
        y_tmp[i] += mat_sh[ind] * x_tmp;
      }
    }

    x_tmp = __ldcg_wrapper(x + (n-1) * strideX);
    #pragma unroll
    for(int i = 0; i < m; i++) {
      const int ind = DG_MAT_IND(i,n-1,m,n);
      y_tmp[i] += mat_sh[ind] * x_tmp;
      y_tmp[i] = alpha * y_tmp[i];
      __stcg_wrapper(y + i * strideY, y_tmp[i]);
    }
  }
}

template<int m, int n>
__global__ void templated_cuda_gemm_gpu_1_beta_sp(
  const int strideX, const int strideY,
  const float alpha,
  const float * __restrict__ matrix,
  const float * __restrict__ arg4, float * __restrict__ arg5,
  const int set_size) {

  extern __shared__ float sh_mem_sp[];

  float *mat_sh = sh_mem_sp;

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = __ldg(matrix + i);
  }
  __syncthreads();

  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if(cell < set_size) {
    const float *x = arg4 + cell;
    float *y = arg5 + cell;

    float y_tmp[m];
    float x_tmp = __ldcg_wrapper(x);
    #pragma unroll
    for(int i = 0; i < m; i++) {
      const int ind = DG_MAT_IND(i,0,m,n);
      y_tmp[i] = mat_sh[ind] * x_tmp;
    }

    #pragma unroll
    for(int j = 1; j < n - 1; j++) {
      x_tmp = __ldcg_wrapper(x + j * strideX);
      #pragma unroll
      for(int i = 0; i < m; i++) {
        const int ind = DG_MAT_IND(i,j,m,n);
        y_tmp[i] += mat_sh[ind] * x_tmp;
      }
    }

    x_tmp = __ldcg_wrapper(x + (n-1) * strideX);
    #pragma unroll
    for(int i = 0; i < m; i++) {
      const int ind = DG_MAT_IND(i,n-1,m,n);
      y_tmp[i] += mat_sh[ind] * x_tmp;
      y_tmp[i] = alpha * y_tmp[i] + __ldcg_wrapper(y + i * strideY);
      __stcg_wrapper(y + i * strideY, y_tmp[i]);
    }
  }
}

/*
 * Transpose kernels
 */

template<int m, int n>
__global__ void templated_cuda_gemm_T_gpu_sp(
  const int strideX, const int strideY,
  const float alpha, const float beta,
  const float * __restrict__ matrix,
  const float * __restrict__ arg4, float * __restrict__ arg5,
  const int set_size) {

  extern __shared__ float sh_mem_sp_t[];

  float *mat_sh = sh_mem_sp_t;

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = __ldg(matrix + i);
  }
  __syncthreads();

  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if(cell < set_size) {
    const float *x = arg4 + cell;
    float *y = arg5 + cell;

    float y_tmp[n];
    float x_tmp = __ldcg_wrapper(x);
    #pragma unroll
    for(int i = 0; i < n; i++) {
      const int ind = DG_MAT_IND(0,i,m,n);
      y_tmp[i] = mat_sh[ind] * x_tmp;
    }

    #pragma unroll
    for(int j = 1; j < m - 1; j++) {
      x_tmp = __ldcg_wrapper(x + j * strideX);
      #pragma unroll
      for(int i = 0; i < n; i++) {
        const int ind = DG_MAT_IND(j,i,m,n);
        y_tmp[i] += mat_sh[ind] * x_tmp;
      }
    }

    x_tmp = __ldcg_wrapper(x + (m-1) * strideX);
    #pragma unroll
    for(int i = 0; i < n; i++) {
      const int ind = DG_MAT_IND(m-1,i,m,n);
      y_tmp[i] += mat_sh[ind] * x_tmp;
      y_tmp[i] = alpha * y_tmp[i] + beta * __ldcg_wrapper(y + i * strideY);
      __stcg_wrapper(y + i * strideY, y_tmp[i]);
    }
  }
}

template<int m, int n>
__global__ void templated_cuda_gemm_T_gpu_1_alpha_0_beta_sp(
  const int strideX, const int strideY,
  const float * __restrict__ matrix,
  const float * __restrict__ arg4, float * __restrict__ arg5,
  const int set_size) {

  extern __shared__ float sh_mem_sp_t[];

  float *mat_sh = sh_mem_sp_t;

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = __ldg(matrix + i);
  }
  __syncthreads();

  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if(cell < set_size) {
    const float *x = arg4 + cell;
    float *y = arg5 + cell;

    float y_tmp[n];
    float x_tmp = __ldcg_wrapper(x);
    #pragma unroll
    for(int i = 0; i < n; i++) {
      const int ind = DG_MAT_IND(0,i,m,n);
      y_tmp[i] = mat_sh[ind] * x_tmp;
    }

    #pragma unroll
    for(int j = 1; j < m - 1; j++) {
      x_tmp = __ldcg_wrapper(x + j * strideX);
      #pragma unroll
      for(int i = 0; i < n; i++) {
        const int ind = DG_MAT_IND(j,i,m,n);
        y_tmp[i] += mat_sh[ind] * x_tmp;
      }
    }

    x_tmp = __ldcg_wrapper(x + (m-1) * strideX);
    #pragma unroll
    for(int i = 0; i < n; i++) {
      const int ind = DG_MAT_IND(m-1,i,m,n);
      y_tmp[i] += mat_sh[ind] * x_tmp;
      __stcg_wrapper(y + i * strideY, y_tmp[i]);
    }
  }
}

template<int m, int n>
__global__ void templated_cuda_gemm_T_gpu_0_beta_sp(
  const int strideX, const int strideY,
  const float alpha,
  const float * __restrict__ matrix,
  const float * __restrict__ arg4, float * __restrict__ arg5,
  const int set_size) {

  extern __shared__ float sh_mem_sp_t[];

  float *mat_sh = sh_mem_sp_t;

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = __ldg(matrix + i);
  }
  __syncthreads();

  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if(cell < set_size) {
    const float *x = arg4 + cell;
    float *y = arg5 + cell;

    float y_tmp[n];
    float x_tmp = __ldcg_wrapper(x);
    #pragma unroll
    for(int i = 0; i < n; i++) {
      const int ind = DG_MAT_IND(0,i,m,n);
      y_tmp[i] = mat_sh[ind] * x_tmp;
    }

    #pragma unroll
    for(int j = 1; j < m - 1; j++) {
      x_tmp = __ldcg_wrapper(x + j * strideX);
      #pragma unroll
      for(int i = 0; i < n; i++) {
        const int ind = DG_MAT_IND(j,i,m,n);
        y_tmp[i] += mat_sh[ind] * x_tmp;
      }
    }

    x_tmp = __ldcg_wrapper(x + (m-1) * strideX);
    #pragma unroll
    for(int i = 0; i < n; i++) {
      const int ind = DG_MAT_IND(m-1,i,m,n);
      y_tmp[i] += mat_sh[ind] * x_tmp;
      y_tmp[i] = alpha * y_tmp[i];
      __stcg_wrapper(y + i * strideY, y_tmp[i]);
    }
  }
}

template<int m, int n>
__global__ void templated_cuda_gemm_T_gpu_1_alpha_1_beta_sp(
  const int strideX, const int strideY,
  const float * __restrict__ matrix,
  const float * __restrict__ arg4, float * __restrict__ arg5,
  const int set_size) {

  extern __shared__ float sh_mem_sp_t[];

  float *mat_sh = sh_mem_sp_t;

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = __ldg(matrix + i);
  }
  __syncthreads();

  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if(cell < set_size) {
    const float *x = arg4 + cell;
    float *y = arg5 + cell;

    float y_tmp[n];
    float x_tmp = __ldcg_wrapper(x);
    #pragma unroll
    for(int i = 0; i < n; i++) {
      const int ind = DG_MAT_IND(0,i,m,n);
      y_tmp[i] = mat_sh[ind] * x_tmp;
    }

    #pragma unroll
    for(int j = 1; j < m - 1; j++) {
      x_tmp = __ldcg_wrapper(x + j * strideX);
      #pragma unroll
      for(int i = 0; i < n; i++) {
        const int ind = DG_MAT_IND(j,i,m,n);
        y_tmp[i] += mat_sh[ind] * x_tmp;
      }
    }

    x_tmp = __ldcg_wrapper(x + (m-1) * strideX);
    #pragma unroll
    for(int i = 0; i < n; i++) {
      const int ind = DG_MAT_IND(m-1,i,m,n);
      y_tmp[i] += mat_sh[ind] * x_tmp;
      y_tmp[i] = y_tmp[i] + __ldcg_wrapper(y + i * strideY);
      __stcg_wrapper(y + i * strideY, y_tmp[i]);
    }
  }
}

template<int m, int n>
__global__ void templated_cuda_gemm_T_gpu_1_beta_sp(
  const int strideX, const int strideY,
  const float alpha,
  const float * __restrict__ matrix,
  const float * __restrict__ arg4, float * __restrict__ arg5,
  const int set_size) {

  extern __shared__ float sh_mem_sp_t[];

  float *mat_sh = sh_mem_sp_t;

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = __ldg(matrix + i);
  }
  __syncthreads();

  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if(cell < set_size) {
    const float *x = arg4 + cell;
    float *y = arg5 + cell;

    float y_tmp[n];
    float x_tmp = __ldcg_wrapper(x);
    #pragma unroll
    for(int i = 0; i < n; i++) {
      const int ind = DG_MAT_IND(0,i,m,n);
      y_tmp[i] = mat_sh[ind] * x_tmp;
    }

    #pragma unroll
    for(int j = 1; j < m - 1; j++) {
      x_tmp = __ldcg_wrapper(x + j * strideX);
      #pragma unroll
      for(int i = 0; i < n; i++) {
        const int ind = DG_MAT_IND(j,i,m,n);
        y_tmp[i] += mat_sh[ind] * x_tmp;
      }
    }

    x_tmp = __ldcg_wrapper(x + (m-1) * strideX);
    #pragma unroll
    for(int i = 0; i < n; i++) {
      const int ind = DG_MAT_IND(m-1,i,m,n);
      y_tmp[i] += mat_sh[ind] * x_tmp;
      y_tmp[i] = alpha * y_tmp[i] + __ldcg_wrapper(y + i * strideY);
      __stcg_wrapper(y + i * strideY, y_tmp[i]);
    }
  }
}

template<int m, int n>
__global__ void templated_cuda_gemm_T_gpu_1_alpha_sp(
  const int strideX, const int strideY,
  const float beta,
  const float * __restrict__ matrix,
  const float * __restrict__ arg4, float * __restrict__ arg5,
  const int set_size) {

  extern __shared__ float sh_mem_sp_t[];

  float *mat_sh = sh_mem_sp_t;

  for(int i = threadIdx.x; i < m * n; i += blockDim.x) {
    mat_sh[i] = __ldg(matrix + i);
  }
  __syncthreads();

  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if(cell < set_size) {
    const float *x = arg4 + cell;
    float *y = arg5 + cell;

    float y_tmp[n];
    float x_tmp = __ldcg_wrapper(x);
    #pragma unroll
    for(int i = 0; i < n; i++) {
      const int ind = DG_MAT_IND(0,i,m,n);
      y_tmp[i] = mat_sh[ind] * x_tmp;
    }

    #pragma unroll
    for(int j = 1; j < m - 1; j++) {
      x_tmp = __ldcg_wrapper(x + j * strideX);
      #pragma unroll
      for(int i = 0; i < n; i++) {
        const int ind = DG_MAT_IND(j,i,m,n);
        y_tmp[i] += mat_sh[ind] * x_tmp;
      }
    }

    x_tmp = __ldcg_wrapper(x + (m-1) * strideX);
    #pragma unroll
    for(int i = 0; i < n; i++) {
      const int ind = DG_MAT_IND(m-1,i,m,n);
      y_tmp[i] += mat_sh[ind] * x_tmp;
      y_tmp[i] = y_tmp[i] + beta * __ldcg_wrapper(y + i * strideY);
      __stcg_wrapper(y + i * strideY, y_tmp[i]);
    }
  }
}
