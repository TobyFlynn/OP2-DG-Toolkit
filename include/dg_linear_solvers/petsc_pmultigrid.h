#ifndef __DG_PETSC_PMULTIGRID_H
#define __DG_PETSC_PMULTIGRID_H

#include "op_seq.h"
#include "linear_solver.h"
#include "petscvec.h"
#include "petscksp.h"
#include "dg_mesh/dg_mesh.h"
#include "pmultigrid.h"

class PETScPMultigrid : public LinearSolver {
public:
  PETScPMultigrid(DGMesh *m);
  ~PETScPMultigrid();

  void init() override;

  void set_coarse_matrix(PoissonCoarseMatrix *c_mat);
  bool solve(op_dat rhs, op_dat ans) override;
  virtual void set_tol_and_iter(const double rtol, const double atol, const int maxiter) override;

  void calc_rhs(Vec in, Vec out);
  void precond(Vec in, Vec out);

private:
  void create_shell_mat();
  void set_shell_pc(PC pc);

  DGMesh *mesh;
  KSP ksp;
  Vec b, x;

  bool pMatInit;
  Mat pMat;
  PMultigridPoissonSolver *pmultigridSolver;
};

#endif
