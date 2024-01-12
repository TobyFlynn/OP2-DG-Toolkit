#ifndef __DG_PETSC_JACOBI_H
#define __DG_PETSC_JACOBI_H

#include "op_seq.h"
#include "linear_solver.h"
#include "petscvec.h"
#include "petscksp.h"
#include "dg_mesh/dg_mesh.h"
#include "dg_matrices/poisson_matrix_free_diag.h"

class PETScJacobiSolver : public LinearSolver {
public:
  PETScJacobiSolver(DGMesh *m);
  ~PETScJacobiSolver();

  void init() override;

  bool solve(op_dat rhs, op_dat ans) override;
  virtual void set_tol_and_iter(const double rtol, const double atol, const int maxiter) override;

  void calc_rhs(Vec in, Vec out);
  void precond(Vec in, Vec out);

private:
  void create_shell_mat();
  void set_shell_pc(PC pc);

  DGMesh *mesh;
  PoissonMatrixFreeDiag *diagMat;
  KSP ksp;
  Vec b, x;
  bool pMatInit, dat_factor;
  Mat pMat;
};

#endif
