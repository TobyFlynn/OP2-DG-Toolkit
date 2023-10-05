#include "dg_linear_solvers/petsc_utils.h"

#include "dg_utils.h"

#include "timing.h"

extern Timing *timer;

#include <stdexcept>

void get_num_nodes_petsc_utils(const int N, int *Np, int *Nfp) {
  #if DG_DIM == 2
  DGUtils::numNodes2D(N, Np, Nfp);
  #elif DG_DIM == 3
  DGUtils::numNodes3D(N, Np, Nfp);
  #endif
}

__global__ void aos_to_soa(const int set_size, const int stride, const DG_FP *in, DG_FP *out) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= set_size * DG_NP) return;
  const int in_ind = tid;
  const int node = tid % DG_NP;
  const int cell = tid / DG_NP;
  const int out_ind = cell + node * stride;
  out[out_ind] = in[in_ind];
}

// Copy PETSc vec array to OP2 dat
void PETScUtils::copy_vec_to_dat(op_dat dat, const DG_FP *dat_d) {
  timer->startTimer("PETScUtils - copy_vec_to_dat");
  op_arg copy_args[] = {
    op_arg_dat(dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE)
  };
  op_mpi_halo_exchanges_grouped(dat->set, 1, copy_args, 2, 0);
  op_mpi_wait_all_grouped(1, copy_args, 2, 0);

  #ifdef DG_OP2_SOA
  const int nthread = 512;
  const int nblocks = dat->set->size * DG_NP / nthread + 1;
  aos_to_soa<<<nblocks,nthread>>>(dat->set->size, getSetSizeFromOpArg(&copy_args[0]), dat_d, (DG_FP *)dat->data_d);
  #else
  cudaMemcpy(dat->data_d, dat_d, dat->set->size * DG_NP * sizeof(DG_FP), cudaMemcpyDeviceToDevice);
  #endif

  op_mpi_set_dirtybit_cuda(1, copy_args);
  timer->endTimer("PETScUtils - copy_vec_to_dat");
}

__global__ void soa_to_aos(const int set_size, const int stride, const DG_FP *in, DG_FP *out) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= set_size * DG_NP) return;
  const int node = tid / set_size;
  const int cell = tid % set_size;
  const int in_ind = cell + node * stride;
  const int out_ind = cell * DG_NP + node;
  out[out_ind] = in[in_ind];
}

// Copy OP2 dat to PETSc vec array
void PETScUtils::copy_dat_to_vec(op_dat dat, DG_FP *dat_d) {
  timer->startTimer("PETScUtils - copy_dat_to_vec");
  op_arg copy_args[] = {
    op_arg_dat(dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ)
  };
  op_mpi_halo_exchanges_grouped(dat->set, 1, copy_args, 2, 0);
  op_mpi_wait_all_grouped(1, copy_args, 2, 0);

  #ifdef DG_OP2_SOA
  const int nthread = 512;
  const int nblocks = dat->set->size * DG_NP / nthread + 1;
  soa_to_aos<<<nblocks,nthread>>>(dat->set->size, getSetSizeFromOpArg(&copy_args[0]), (DG_FP *)dat->data_d, dat_d);
  #else
  cudaMemcpy(dat_d, dat->data_d, dat->set->size * DG_NP * sizeof(DG_FP), cudaMemcpyDeviceToDevice);
  #endif

  op_mpi_set_dirtybit_cuda(1, copy_args);
  timer->endTimer("PETScUtils - copy_dat_to_vec");
}

// Create a PETSc vector for GPUs
void PETScUtils::create_vec(Vec *v, op_set set) {
  timer->startTimer("PETScUtils - create_vec");
  VecCreate(PETSC_COMM_WORLD, v);
  VecSetType(*v, VECCUDA);
  VecSetSizes(*v, set->size * DG_NP, PETSC_DECIDE);
  timer->endTimer("PETScUtils - create_vec");
}

// Destroy a PETSc vector
void PETScUtils::destroy_vec(Vec *v) {
  timer->startTimer("PETScUtils - destroy_vec");
  VecDestroy(v);
  timer->endTimer("PETScUtils - destroy_vec");
}

// Load a PETSc vector with values from an OP2 dat for GPUs
void PETScUtils::load_vec(Vec *v, op_dat v_dat) {
  timer->startTimer("PETScUtils - load_vec");
  DG_FP *v_ptr;
  VecCUDAGetArray(*v, &v_ptr);

  copy_dat_to_vec(v_dat, v_ptr);

  VecCUDARestoreArray(*v, &v_ptr);
  timer->endTimer("PETScUtils - load_vec");
}

// Load an OP2 dat with the values from a PETSc vector for GPUs
void PETScUtils::store_vec(Vec *v, op_dat v_dat) {
  timer->startTimer("PETScUtils - store_vec");
  const DG_FP *v_ptr;
  VecCUDAGetArrayRead(*v, &v_ptr);

  copy_vec_to_dat(v_dat, v_ptr);

  VecCUDARestoreArrayRead(*v, &v_ptr);
  timer->endTimer("PETScUtils - store_vec");
}

void PETScUtils::create_vec_coarse(Vec *v, op_set set) {
  timer->startTimer("PETScUtils - create_vec");
  VecCreate(PETSC_COMM_WORLD, v);
  VecSetType(*v, VECCUDA);
  VecSetSizes(*v, set->size * DG_NP_N1, PETSC_DECIDE);
  timer->endTimer("PETScUtils - create_vec");
}

__global__ void aos_to_soa_coarse(const int set_size, const int stride, const DG_FP *in, float *out) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= set_size * DG_NP_N1) return;
  const int in_ind = tid;
  const int node = tid % DG_NP_N1;
  const int cell = tid / DG_NP_N1;
  const int out_ind = cell + node * stride;
  out[out_ind] = (float)in[in_ind];
}

// Copy PETSc vec array to OP2 dat
void PETScUtils::copy_vec_to_dat_coarse(op_dat dat, const DG_FP *dat_d) {
  timer->startTimer("PETScUtils - copy_vec_to_dat_coarse");
  op_arg copy_args[] = {
    op_arg_dat(dat, -1, OP_ID, DG_NP, "float", OP_WRITE)
  };
  op_mpi_halo_exchanges_grouped(dat->set, 1, copy_args, 2, 0);
  op_mpi_wait_all_grouped(1, copy_args, 2, 0);

  #ifdef DG_OP2_SOA
  const int nthread = 512;
  const int nblocks = dat->set->size * DG_NP_N1 / nthread + 1;
  aos_to_soa_coarse<<<nblocks,nthread>>>(dat->set->size, getSetSizeFromOpArg(&copy_args[0]), dat_d, (float *)dat->data_d);
  #else
  cudaMemcpy2D(dat->data_d, DG_NP * sizeof(DG_FP), dat_d, DG_NP_N1 * sizeof(DG_FP), DG_NP_N1 * sizeof(DG_FP), dat->set->size, cudaMemcpyDeviceToDevice);
  #endif

  op_mpi_set_dirtybit_cuda(1, copy_args);
  timer->endTimer("PETScUtils - copy_vec_to_dat_coarse");
}

__global__ void soa_to_aos_coarse(const int set_size, const int stride, const float *in, DG_FP *out) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= set_size * DG_NP_N1) return;
  const int node = tid / set_size;
  const int cell = tid % set_size;
  const int in_ind = cell + node * stride;
  const int out_ind = cell * DG_NP_N1 + node;
  out[out_ind] = (double)in[in_ind];
}

// Copy OP2 dat to PETSc vec array
void PETScUtils::copy_dat_to_vec_coarse(op_dat dat, DG_FP *dat_d) {
  timer->startTimer("PETScUtils - copy_dat_to_vec");
  op_arg copy_args[] = {
    op_arg_dat(dat, -1, OP_ID, DG_NP, "float", OP_READ)
  };
  op_mpi_halo_exchanges_grouped(dat->set, 1, copy_args, 2, 0);
  op_mpi_wait_all_grouped(1, copy_args, 2, 0);

  #ifdef DG_OP2_SOA
  const int nthread = 512;
  const int nblocks = dat->set->size * DG_NP_N1 / nthread + 1;
  soa_to_aos_coarse<<<nblocks,nthread>>>(dat->set->size, getSetSizeFromOpArg(&copy_args[0]), (float *)dat->data_d, dat_d);
  #else
  cudaMemcpy2D(dat_d, DG_NP_N1 * sizeof(DG_FP), dat->data_d, DG_NP * sizeof(DG_FP), DG_NP_N1 * sizeof(DG_FP), dat->set->size, cudaMemcpyDeviceToDevice);
  #endif

  op_mpi_set_dirtybit_cuda(1, copy_args);
  timer->endTimer("PETScUtils - copy_dat_to_vec");
}

// Load a PETSc vector with values from an OP2 dat for CPUs
void PETScUtils::load_vec_coarse(Vec *v, op_dat v_dat) {
  timer->startTimer("PETScUtils - load_vec_coarse");
  DG_FP *v_ptr;
  VecCUDAGetArray(*v, &v_ptr);

  copy_dat_to_vec_coarse(v_dat, v_ptr);

  VecCUDARestoreArray(*v, &v_ptr);
  timer->endTimer("PETScUtils - load_vec_coarse");
}

// Load an OP2 dat with the values from a PETSc vector for CPUs
void PETScUtils::store_vec_coarse(Vec *v, op_dat v_dat) {
  timer->startTimer("PETScUtils - store_vec_coarse");
  const DG_FP *v_ptr;
  VecCUDAGetArrayRead(*v, &v_ptr);

  copy_vec_to_dat_coarse(v_dat, v_ptr);

  VecCUDARestoreArrayRead(*v, &v_ptr);
  timer->endTimer("PETScUtils - store_vec_coarse");
}
