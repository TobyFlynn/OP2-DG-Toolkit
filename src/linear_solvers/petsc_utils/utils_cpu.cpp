#include "dg_linear_solvers/petsc_utils.h"

#include "dg_utils.h"

#include "timing.h"

extern Timing *timer;

void get_num_nodes_petsc_utils(const int N, int *Np, int *Nfp) {
  #if DG_DIM == 2
  DGUtils::numNodes2D(N, Np, Nfp);
  #elif DG_DIM == 3
  DGUtils::numNodes3D(N, Np, Nfp);
  #endif
}

// Copy PETSc vec array to OP2 dat
void PETScUtils::copy_vec_to_dat(op_dat dat, const DG_FP *dat_d) {
  timer->startTimer("PETScUtils - copy_vec_to_dat");
  op_arg copy_args[] = {
    op_arg_dat(dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE)
  };
  op_mpi_halo_exchanges(dat->set, 1, copy_args);

  memcpy(dat->data, dat_d, dat->set->size * DG_NP * sizeof(DG_FP));

  op_mpi_set_dirtybit(1, copy_args);
  timer->endTimer("PETScUtils - copy_vec_to_dat");
}

// Copy OP2 dat to PETSc vec array
void PETScUtils::copy_dat_to_vec(op_dat dat, DG_FP *dat_d) {
  timer->startTimer("PETScUtils - copy_dat_to_vec");
  op_arg copy_args[] = {
    op_arg_dat(dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ)
  };
  op_mpi_halo_exchanges(dat->set, 1, copy_args);

  memcpy(dat_d, dat->data, dat->set->size * DG_NP * sizeof(DG_FP));

  op_mpi_set_dirtybit(1, copy_args);
  timer->endTimer("PETScUtils - copy_dat_to_vec");
}

// Create a PETSc vector for CPUs
void PETScUtils::create_vec(Vec *v, op_set set) {
  timer->startTimer("PETScUtils - create_vec");
  VecCreate(PETSC_COMM_WORLD, v);
  VecSetType(*v, VECSTANDARD);
  VecSetSizes(*v, set->size * DG_NP, PETSC_DECIDE);
  timer->endTimer("PETScUtils - create_vec");
}

// Destroy a PETSc vector
void PETScUtils::destroy_vec(Vec *v) {
  timer->startTimer("PETScUtils - destroy_vec");
  VecDestroy(v);
  timer->endTimer("PETScUtils - destroy_vec");
}

// Load a PETSc vector with values from an OP2 dat for CPUs
void PETScUtils::load_vec(Vec *v, op_dat v_dat) {
  timer->startTimer("PETScUtils - load_vec");
  DG_FP *v_ptr;
  VecGetArray(*v, &v_ptr);

  copy_dat_to_vec(v_dat, v_ptr);

  VecRestoreArray(*v, &v_ptr);
  timer->endTimer("PETScUtils - load_vec");
}

// Load an OP2 dat with the values from a PETSc vector for CPUs
void PETScUtils::store_vec(Vec *v, op_dat v_dat) {
  timer->startTimer("PETScUtils - store_vec");
  const DG_FP *v_ptr;
  VecGetArrayRead(*v, &v_ptr);

  copy_vec_to_dat(v_dat, v_ptr);

  VecRestoreArrayRead(*v, &v_ptr);
  timer->endTimer("PETScUtils - store_vec");
}

void PETScUtils::create_vec_coarse(Vec *v, op_set set) {
  timer->startTimer("PETScUtils - create_vec");
  VecCreate(PETSC_COMM_WORLD, v);
  VecSetType(*v, VECSTANDARD);
  VecSetSizes(*v, set->size * DG_NP_N1, PETSC_DECIDE);
  timer->endTimer("PETScUtils - create_vec");
}

// Copy PETSc vec array to OP2 dat
void PETScUtils::copy_vec_to_dat_coarse(op_dat dat, const DG_FP *dat_d) {
  timer->startTimer("PETScUtils - copy_vec_to_dat_coarse");
  op_arg copy_args[] = {
    op_arg_dat(dat, -1, OP_ID, DG_NP, "float", OP_WRITE)
  };
  op_mpi_halo_exchanges(dat->set, 1, copy_args);

  float *op2_dat_ptr = (float *)dat->data;

  #pragma omp parallel for
  for(int i = 0; i < dat->set->size; i++) {
    #pragma unroll
    for(int j = 0; j < DG_NP_N1; j++) {
      op2_dat_ptr[i * DG_NP + j] = (float)dat_d[i * DG_NP_N1 + j];
    }
  }

  op_mpi_set_dirtybit(1, copy_args);
  timer->endTimer("PETScUtils - copy_vec_to_dat_coarse");
}

// Copy OP2 dat to PETSc vec array
void PETScUtils::copy_dat_to_vec_coarse(op_dat dat, DG_FP *dat_d) {
  timer->startTimer("PETScUtils - copy_dat_to_vec_coarse");
  op_arg copy_args[] = {
    op_arg_dat(dat, -1, OP_ID, DG_NP, "float", OP_READ)
  };
  op_mpi_halo_exchanges(dat->set, 1, copy_args);

  float *op2_dat_ptr = (float *)dat->data;

  #pragma omp parallel for
  for(int i = 0; i < dat->set->size; i++) {
    #pragma unroll
    for(int j = 0; j < DG_NP_N1; j++) {
      dat_d[i * DG_NP_N1 + j] = (double)op2_dat_ptr[i * DG_NP + j];
    }
  }

  op_mpi_set_dirtybit(1, copy_args);
  timer->endTimer("PETScUtils - copy_dat_to_vec_coarse");
}

// Load a PETSc vector with values from an OP2 dat for CPUs
void PETScUtils::load_vec_coarse(Vec *v, op_dat v_dat) {
  timer->startTimer("PETScUtils - load_vec_coarse");
  DG_FP *v_ptr;
  VecGetArray(*v, &v_ptr);

  copy_dat_to_vec_coarse(v_dat, v_ptr);

  VecRestoreArray(*v, &v_ptr);
  timer->endTimer("PETScUtils - load_vec_coarse");
}

// Load an OP2 dat with the values from a PETSc vector for CPUs
void PETScUtils::store_vec_coarse(Vec *v, op_dat v_dat) {
  timer->startTimer("PETScUtils - store_vec_coarse");
  const DG_FP *v_ptr;
  VecGetArrayRead(*v, &v_ptr);

  copy_vec_to_dat_coarse(v_dat, v_ptr);

  VecRestoreArrayRead(*v, &v_ptr);
  timer->endTimer("PETScUtils - store_vec_coarse");
}
