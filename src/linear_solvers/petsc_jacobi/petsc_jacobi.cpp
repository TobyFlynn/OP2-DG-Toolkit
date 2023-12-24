#include "dg_linear_solvers/petsc_jacobi.h"

#include "op_seq.h"

#include "dg_constants/dg_constants.h"

#include "timing.h"

#include "op2_utils.h"
#include "dg_linear_solvers/petsc_utils.h"
#include "timing.h"
#include "config.h"
#include "dg_dat_pool.h"
#include "dg_abort.h"

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <type_traits>

extern DGConstants *constants;
extern Timing *timer;
extern DGDatPool *dg_dat_pool;
extern Config *config;

PETScJacobiSolver::PETScJacobiSolver(DGMesh *m) {
  bc = nullptr;
  nullspace = false;
  pMatInit = false;
  mesh = m;

  KSPCreate(PETSC_COMM_WORLD, &ksp);
  // KSPSetType(ksp, KSPGMRES);
  KSPSetType(ksp, KSPCG);
  double r_tol, a_tol;
  if(std::is_same<DG_FP,double>::value) {
    r_tol = 1e-8;
    a_tol = 1e-9;
  } else {
    r_tol = 1e-5;
    a_tol = 1e-6;
  }
  KSPSetTolerances(ksp, r_tol, a_tol, 1e5, 5e2);
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCSHELL);
  set_shell_pc(pc);
}

PETScJacobiSolver::~PETScJacobiSolver() {
  if(pMatInit)
    MatDestroy(&pMat);
  KSPDestroy(&ksp);
  PETScUtils::destroy_vec(&b);
  PETScUtils::destroy_vec(&x);
}

void PETScJacobiSolver::init() {
  PETScUtils::create_vec(&b, mesh->cells);
  PETScUtils::create_vec(&x, mesh->cells);
}

bool PETScJacobiSolver::solve(op_dat rhs, op_dat ans) {
  timer->startTimer("PETScJacobiSolver - solve");

  if(dynamic_cast<PoissonMatrixFreeDiag*>(matrix) == nullptr) {
    dg_abort("PETScJacobiSolver matrix should be of type PoissonMatrixFreeDiag\n");
  }
  diagMat = dynamic_cast<PoissonMatrixFreeDiag*>(matrix);

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

  if(bc)
    matrix->apply_bc(rhs, bc);

  // PETScUtils::load_vec_p_adapt(&b, rhs, mesh);
  // PETScUtils::load_vec_p_adapt(&x, ans, mesh);
  PETScUtils::load_vec(&b, rhs);
  PETScUtils::load_vec(&x, ans);

  timer->startTimer("PETScJacobiSolver - KSPSolve");
  KSPSolve(ksp, b, x);
  timer->endTimer("PETScJacobiSolver - KSPSolve");

  PetscInt numIt;
  KSPGetIterationNumber(ksp, &numIt);
  KSPConvergedReason reason;
  KSPGetConvergedReason(ksp, &reason);
  // Check that the solver converged
  bool converged = true;
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

  timer->endTimer("PETScJacobiSolver - solve");

  return converged;
}

void PETScJacobiSolver::calc_rhs(Vec in, Vec out) {
  timer->startTimer("PETScJacobiSolver - calc_rhs");
  // Copy u to OP2 dat
  DGTempDat tmp_in  = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_out = dg_dat_pool->requestTempDatCells(DG_NP);
  PETScUtils::store_vec(&in, tmp_in.dat);

  matrix->mult(tmp_in.dat, tmp_out.dat);

  PETScUtils::load_vec(&out, tmp_out.dat);
  dg_dat_pool->releaseTempDatCells(tmp_in);
  dg_dat_pool->releaseTempDatCells(tmp_out);
  timer->endTimer("PETScJacobiSolver - calc_rhs");
}

// Matrix-free inv Mass preconditioning function
void PETScJacobiSolver::precond(Vec in, Vec out) {
  timer->startTimer("PETScJacobiSolver - precond");
  DGTempDat tmp_in  = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_out = dg_dat_pool->requestTempDatCells(DG_NP);
  PETScUtils::store_vec(&in, tmp_in.dat);

  op_par_loop(petsc_pre_jacobi, "petsc_pre_jacobi", mesh->cells,
              op_arg_dat(diagMat->diag, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_in.dat,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_out.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  PETScUtils::load_vec(&out, tmp_out.dat);
  dg_dat_pool->releaseTempDatCells(tmp_in);
  dg_dat_pool->releaseTempDatCells(tmp_out);
  timer->endTimer("PETScJacobiSolver - precond");
}

PetscErrorCode matMultPJS(Mat A, Vec x, Vec y) {
  PETScJacobiSolver *solver;
  MatShellGetContext(A, &solver);
  solver->calc_rhs(x, y);
  return 0;
}

void PETScJacobiSolver::create_shell_mat() {
  if(pMatInit)
    MatDestroy(&pMat);

  MatCreateShell(PETSC_COMM_WORLD, matrix->getUnknowns(), matrix->getUnknowns(), PETSC_DETERMINE, PETSC_DETERMINE, this, &pMat);
  MatShellSetOperation(pMat, MATOP_MULT, (void(*)(void))matMultPJS);

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

PetscErrorCode preconPJS(PC pc, Vec x, Vec y) {
  PETScJacobiSolver *solver;
  PCShellGetContext(pc, (void **)&solver);
  solver->precond(x, y);
  return 0;
}

void PETScJacobiSolver::set_shell_pc(PC pc) {
  PCShellSetApply(pc, preconPJS);
  PCShellSetContext(pc, this);
}

void PETScJacobiSolver::set_tol_and_iter(const double rtol, const double atol, const int maxiter) {
  KSPSetTolerances(ksp, rtol, atol, 1e5, maxiter);
}