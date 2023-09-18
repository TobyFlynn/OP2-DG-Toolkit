#include "dg_linear_solvers/petsc_pmultigrid.h"

#ifdef DG_MPI
#include "mpi_helper_func.h"
#endif

/*
PetscErrorCode matAMultPM(Mat A, Vec x, Vec y) {
  PETScPMultigrid *poisson;
  MatShellGetContext(A, &poisson);
  const DG_FP *x_ptr;
  DG_FP *y_ptr;
  VecCUDAGetArrayRead(x, &x_ptr);
  VecCUDAGetArray(y, &y_ptr);

  poisson->calc_rhs(x_ptr, y_ptr);

  VecCUDARestoreArrayRead(x, &x_ptr);
  VecCUDARestoreArray(y, &y_ptr);
  return 0;
}
*/

void PETScPMultigrid::create_shell_mat() {
  throw std::runtime_error("PETScPMultigrid::create_shell_mat not implemented for HIP yet");
  /*
  if(pMatInit)
    MatDestroy(&pMat);

  // const int mat_size = matrix->getUnknowns();
  const int mat_size = mesh->cells->size * DG_NP;
  MatCreateShell(PETSC_COMM_WORLD, mat_size, mat_size, PETSC_DETERMINE, PETSC_DETERMINE, this, &pMat);
  MatShellSetOperation(pMat, MATOP_MULT, (void(*)(void))matAMultPM);
  MatShellSetVecType(pMat, VECCUDA);

  pMatInit = true;
  */
}

/*
PetscErrorCode preconPM(PC pc, Vec x, Vec y) {
  PETScPMultigrid *poisson;
  PCShellGetContext(pc, (void **)&poisson);
  const DG_FP *x_ptr;
  DG_FP *y_ptr;
  VecCUDAGetArrayRead(x, &x_ptr);
  VecCUDAGetArray(y, &y_ptr);

  poisson->precond(x_ptr, y_ptr);

  VecCUDARestoreArrayRead(x, &x_ptr);
  VecCUDARestoreArray(y, &y_ptr);
  return 0;
}
*/

void PETScPMultigrid::set_shell_pc(PC pc) {
  throw std::runtime_error("PETScPMultigrid::set_shell_pc not implemented for HIP yet");
  /*
  PCShellSetApply(pc, preconPM);
  PCShellSetContext(pc, this);
  */
}
