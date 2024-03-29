#ifndef __DG_PETSC_INV_MASS_H
#define __DG_PETSC_INV_MASS_H

#include "op_seq.h"
#include "linear_solver.h"
#include "petscvec.h"
#include "petscksp.h"
#include "dg_mesh/dg_mesh.h"

class PETScInvMassSolver : public LinearSolver {
public:
  PETScInvMassSolver(DGMesh *m);
  ~PETScInvMassSolver();

  bool solve(op_dat rhs, op_dat ans) override;
  virtual void set_tol_and_iter(const double rtol, const double atol, const int maxiter) override;

  void calc_rhs(Vec in, Vec out);
  void precond(Vec in, Vec out);
  void setFactor(const double f);
  void setFactor(op_dat f);

private:
  void calc_precond_mat();
  void create_shell_mat();
  void set_shell_pc(PC pc);

  DGMesh *mesh;
  KSP ksp;
  op_dat factor_dat;
  bool pMatInit, dat_factor;
  Mat pMat;
  DG_FP factor;
};

#endif
