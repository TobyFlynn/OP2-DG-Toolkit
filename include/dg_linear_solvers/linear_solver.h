#ifndef __DG_LINEAR_SOLVER_H
#define __DG_LINEAR_SOLVER_H

#include "dg_compiler_defs.h"

#include "op_seq.h"
#include "dg_matrices/poisson_matrix.h"

#include <vector>

class LinearSolver {
public:
  enum Solvers {
    AMGX_AMG, HYPRE_AMG, PETSC_AMG, PETSC_AMG_COARSE,
    PETSC_JACOBI, PETSC_BLOCK_JACOBI, PETSC_INV_MASS,
    PETSC_PMULTIGRID, PMULTIGRID
  };

  virtual void set_matrix(PoissonMatrix *mat);
  void set_bcs(op_dat bcs);
  void set_nullspace(bool ns);
  virtual bool solve(op_dat rhs, op_dat ans) = 0;
  virtual void init();
  virtual void set_tol_and_iter(const double rtol, const double atol, const int maxiter) = 0;

protected:
  PoissonMatrix *matrix;
  bool nullspace;
  op_dat bc;
  std::vector<int> iter_counts;
};

#endif
