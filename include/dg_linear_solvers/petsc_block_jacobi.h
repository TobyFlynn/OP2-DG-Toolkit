#ifndef __DG_PETSC_BLOCK_JACOBI_H
#define __DG_PETSC_BLOCK_JACOBI_H

#include "op_seq.h"
#include "linear_solver.h"
#include "petscvec.h"
#include "petscksp.h"
#include "dg_mesh/dg_mesh.h"

class PETScBlockJacobiSolver : public LinearSolver {
public:
  PETScBlockJacobiSolver(DGMesh *m);
  ~PETScBlockJacobiSolver();

  bool solve(op_dat rhs, op_dat ans) override;

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
};

#endif
