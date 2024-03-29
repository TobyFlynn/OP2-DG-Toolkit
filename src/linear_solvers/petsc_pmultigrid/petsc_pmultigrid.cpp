#include "dg_linear_solvers/petsc_pmultigrid.h"

#include "op_seq.h"

#include <iostream>
#include <type_traits>

#include "dg_linear_solvers/petsc_utils.h"
#include "timing.h"
#include "config.h"
#include "dg_dat_pool.h"

extern Timing *timer;
extern DGDatPool *dg_dat_pool;
extern Config *config;

PETScPMultigrid::PETScPMultigrid(DGMesh *m) {
  bc = nullptr;
  mesh = m;
  nullspace = false;
  pMatInit = false;

  pmultigridSolver = new PMultigridPoissonSolver(mesh);

  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetType(ksp, KSPFCG);
  double r_tol, a_tol;
  if(std::is_same<DG_FP,double>::value) {
    r_tol = 1e-8;
    a_tol = 1e-9;
  } else {
    r_tol = 1e-5;
    a_tol = 1e-6;
  }
  KSPSetTolerances(ksp, r_tol, a_tol, 1e5, 2.5e2);
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCSHELL);
  set_shell_pc(pc);
}

PETScPMultigrid::~PETScPMultigrid() {
  op_printf("\n");
  for(int i = 0; i < iter_counts.size(); i++) {
    op_printf("%d,", iter_counts[i]);
  }
  op_printf("\n");
  PETScUtils::destroy_vec(&b);
  PETScUtils::destroy_vec(&x);
  if(pMatInit)
    MatDestroy(&pMat);
  KSPDestroy(&ksp);
  delete pmultigridSolver;
}

void PETScPMultigrid::init() {
  PETScUtils::create_vec(&b, mesh->cells);
  PETScUtils::create_vec(&x, mesh->cells);
  pmultigridSolver->init();
}

void PETScPMultigrid::set_coarse_matrix(PoissonCoarseMatrix *c_mat) {
  pmultigridSolver->set_coarse_matrix(c_mat);
}

bool PETScPMultigrid::solve(op_dat rhs, op_dat ans) {
  timer->startTimer("PETScPMultigrid - solve");
  if(!pMatInit)
    create_shell_mat();
  if(nullspace) {
    MatNullSpace ns;
    MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &ns);
    MatSetNullSpace(pMat, ns);
    MatSetTransposeNullSpace(pMat, ns);
    MatNullSpaceDestroy(&ns);
  }
  MatAssemblyBegin(pMat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(pMat, MAT_FINAL_ASSEMBLY);
  KSPSetOperators(ksp, pMat, pMat);

  pmultigridSolver->set_matrix(matrix);
  pmultigridSolver->set_nullspace(nullspace);

  if(bc)
    matrix->apply_bc(rhs, bc);

  // PETScUtils::load_vec_p_adapt(&b, rhs, mesh);
  // PETScUtils::load_vec_p_adapt(&x, ans, mesh);
  PETScUtils::load_vec(&b, rhs);
  PETScUtils::load_vec(&x, ans);

  timer->startTimer("PETScPMultigrid - KSPSolve");
  KSPSolve(ksp, b, x);
  timer->endTimer("PETScPMultigrid - KSPSolve");

  PetscInt numIt;
  KSPGetIterationNumber(ksp, &numIt);
  iter_counts.push_back(numIt);
  // op_printf("%d\n", numIt);
  KSPConvergedReason reason;
  KSPGetConvergedReason(ksp, &reason);
  // Check that the solver converged
  bool converged = true;
  // std::cout << "Number of iterations for PETSc PMultigrid: " << numIt << std::endl;
  if(reason < 0) {
    DG_FP residual;
    KSPGetResidualNorm(ksp, &residual);
    converged = false;
    std::cout << "Number of iterations for linear solver: " << numIt << std::endl;
    std::cout << "Converged reason: " << reason << " Residual: " << residual << std::endl;
  }

  Vec solution;
  KSPGetSolution(ksp, &solution);
  // PETScUtils::store_vec_p_adapt(&solution, ans, mesh);
  PETScUtils::store_vec(&solution, ans);

  timer->endTimer("PETScPMultigrid - solve");

  return converged;
}

void PETScPMultigrid::calc_rhs(Vec in, Vec out) {
  timer->startTimer("PETScPMultigrid - calc_rhs");
  // Copy u to OP2 dat
  DGTempDat tmp_in  = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_out = dg_dat_pool->requestTempDatCells(DG_NP);
  PETScUtils::store_vec(&in, tmp_in.dat);

  matrix->mult(tmp_in.dat, tmp_out.dat);

  PETScUtils::load_vec(&out, tmp_out.dat);
  dg_dat_pool->releaseTempDatCells(tmp_in);
  dg_dat_pool->releaseTempDatCells(tmp_out);
  timer->endTimer("PETScPMultigrid - calc_rhs");
}

void PETScPMultigrid::precond(Vec in, Vec out) {
  timer->startTimer("PETScPMultigrid - precond");
  DGTempDat tmp_in  = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_out = dg_dat_pool->requestTempDatCells(DG_NP);
  PETScUtils::store_vec(&in, tmp_in.dat);

  pmultigridSolver->solve(tmp_in.dat, tmp_out.dat);

  PETScUtils::load_vec(&out, tmp_out.dat);
  dg_dat_pool->releaseTempDatCells(tmp_in);
  dg_dat_pool->releaseTempDatCells(tmp_out);
  timer->endTimer("PETScPMultigrid - precond");
}

PetscErrorCode matMultPPMS(Mat A, Vec x, Vec y) {
  PETScPMultigrid *solver;
  MatShellGetContext(A, &solver);
  solver->calc_rhs(x, y);
  return 0;
}

void PETScPMultigrid::create_shell_mat() {
  if(pMatInit)
    MatDestroy(&pMat);

  MatCreateShell(PETSC_COMM_WORLD, matrix->getUnknowns(), matrix->getUnknowns(), PETSC_DETERMINE, PETSC_DETERMINE, this, &pMat);
  MatShellSetOperation(pMat, MATOP_MULT, (void(*)(void))matMultPPMS);

  #if defined(OP2_DG_CUDA)
  MatShellSetVecType(pMat, VECCUDA);
  #elif defined(OP2_DG_HIP)
  #ifdef PETSC_COMPILED_WITH_HIP
  MatShellSetVecType(pMat, VECHIP);
  #else
  MatShellSetVecType(pMat, VECSTANDARD);
  #endif
  #else
  MatShellSetVecType(pMat, VECSTANDARD);
  #endif

  pMatInit = true;
}

PetscErrorCode preconPPMS(PC pc, Vec x, Vec y) {
  PETScPMultigrid *solver;
  PCShellGetContext(pc, (void **)&solver);
  solver->precond(x, y);
  return 0;
}

void PETScPMultigrid::set_shell_pc(PC pc) {
  PCShellSetApply(pc, preconPPMS);
  PCShellSetContext(pc, this);
}

void PETScPMultigrid::set_tol_and_iter(const double rtol, const double atol, const int maxiter) {
  KSPSetTolerances(ksp, rtol, atol, 1e5, maxiter);
}