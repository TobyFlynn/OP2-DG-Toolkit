#ifndef __DG_PETSC_BLOCK_JACOBI_H
#define __DG_PETSC_BLOCK_JACOBI_H

#include "op_seq.h"
#include "linear_solver.h"
#include "petscvec.h"
#include "petscksp.h"
#include "dg_mesh/dg_mesh.h"
#include "dg_matrices/poisson_matrix_free_block_diag.h"

class PETScBlockJacobiSolver : public LinearSolver {
public:
  PETScBlockJacobiSolver(DGMesh *m);
  ~PETScBlockJacobiSolver();

  void set_matrix(PoissonMatrix *mat) override;
  bool solve(op_dat rhs, op_dat ans) override;
  virtual void set_tol_and_iter(const double rtol, const double atol, const int maxiter) override;

  void calc_rhs(Vec in, Vec out);
  void precond(Vec in, Vec out);

private:
  void calc_precond_mat();
  void create_shell_mat();
  void set_shell_pc(PC pc);

  DGMesh *mesh;
  KSP ksp;
  op_dat pre;
  bool pMatInit;
  Mat pMat;
  PoissonMatrixFreeBlockDiag *block_matrix;
};

#endif
