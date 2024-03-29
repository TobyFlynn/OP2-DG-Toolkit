#include "dg_linear_solvers/petsc_inv_mass.h"

#include "op_seq.h"

#include "dg_constants/dg_constants.h"

#include "timing.h"

#include "op2_utils.h"
#include "dg_linear_solvers/petsc_utils.h"
#include "timing.h"
#include "config.h"
#include "dg_dat_pool.h"

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <type_traits>

extern DGConstants *constants;
extern Config *config;
extern Timing *timer;
extern DGDatPool *dg_dat_pool;

PETScInvMassSolver::PETScInvMassSolver(DGMesh *m) {
  bc = nullptr;
  nullspace = false;
  pMatInit = false;
  mesh = m;
  factor = 1.0;
  dat_factor = false;

  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetType(ksp, KSPGMRES);
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

PETScInvMassSolver::~PETScInvMassSolver() {
  MatDestroy(&pMat);
  KSPDestroy(&ksp);
}

bool PETScInvMassSolver::solve(op_dat rhs, op_dat ans) {
  timer->startTimer("PETScInvMassSolver - solve");
  if(!pMatInit)
    create_shell_mat();
  KSPSetOperators(ksp, pMat, pMat);
  if(nullspace) {
    MatNullSpace ns;
    MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &ns);
    MatSetNullSpace(pMat, ns);
    MatSetTransposeNullSpace(pMat, ns);
    MatNullSpaceDestroy(&ns);
  }

  if(bc)
    matrix->apply_bc(rhs, bc);

  Vec b, x;
  PETScUtils::create_vec(&b, mesh->cells);
  PETScUtils::create_vec(&x, mesh->cells);

  PETScUtils::load_vec(&b, rhs);
  PETScUtils::load_vec(&x, ans);

  timer->startTimer("PETScInvMassSolver - KSPSolve");
  KSPSolve(ksp, b, x);
  timer->endTimer("PETScInvMassSolver - KSPSolve");

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
  PETScUtils::store_vec(&solution, ans);

  PETScUtils::destroy_vec(&b);
  PETScUtils::destroy_vec(&x);

  timer->endTimer("PETScInvMassSolver - solve");

  return converged;
}

void PETScInvMassSolver::calc_rhs(Vec in, Vec out) {
  timer->startTimer("PETScInvMassSolver - calc_rhs");
  // Copy u to OP2 dat
  DGTempDat tmp_in  = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_out = dg_dat_pool->requestTempDatCells(DG_NP);
  PETScUtils::store_vec(&in, tmp_in.dat);

  matrix->mult(tmp_in.dat, tmp_out.dat);

  PETScUtils::load_vec(&out, tmp_out.dat);
  dg_dat_pool->releaseTempDatCells(tmp_in);
  dg_dat_pool->releaseTempDatCells(tmp_out);
  timer->endTimer("PETScInvMassSolver - calc_rhs");
}

// Matrix-free inv Mass preconditioning function
void PETScInvMassSolver::precond(Vec in, Vec out) {
  timer->startTimer("PETScInvMassSolver - precond");
  DGTempDat tmp_in  = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_out = dg_dat_pool->requestTempDatCells(DG_NP);
  PETScUtils::store_vec(&in, tmp_in.dat);

  #if DG_DIM == 3
  if(dat_factor) {
    op_par_loop(petsc_pre_inv_mass_dat, "petsc_pre_inv_mass_dat", mesh->cells,
                op_arg_dat(mesh->geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
                op_arg_dat(factor_dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(tmp_in.dat,      -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(tmp_out.dat,     -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));
  } else {
    op_par_loop(petsc_pre_inv_mass, "petsc_pre_inv_mass", mesh->cells,
                op_arg_gbl(&factor,  1, DG_FP_STR, OP_READ),
                op_arg_dat(mesh->geof, -1, OP_ID, 10, DG_FP_STR, OP_READ),
                op_arg_dat(tmp_in.dat,      -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
                op_arg_dat(tmp_out.dat,     -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));
  }
  #else
  op_par_loop(petsc_pre_inv_mass_2d, "petsc_pre_inv_mass_2d", mesh->cells,
              op_arg_gbl(&factor,  1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->geof, -1, OP_ID, 5, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_in.dat,      -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_out.dat,     -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));
  #endif

  PETScUtils::load_vec(&out, tmp_out.dat);
  dg_dat_pool->releaseTempDatCells(tmp_in);
  dg_dat_pool->releaseTempDatCells(tmp_out);
  timer->endTimer("PETScInvMassSolver - precond");
}

void PETScInvMassSolver::setFactor(const double f) {
  factor = f;
}

void PETScInvMassSolver::setFactor(op_dat f) {
  factor_dat = f;
  dat_factor = true;
}

PetscErrorCode matMultPIMS(Mat A, Vec x, Vec y) {
  PETScInvMassSolver *solver;
  MatShellGetContext(A, &solver);
  solver->calc_rhs(x, y);
  return 0;
}

void PETScInvMassSolver::create_shell_mat() {
  if(pMatInit)
    MatDestroy(&pMat);

  MatCreateShell(PETSC_COMM_WORLD, matrix->getUnknowns(), matrix->getUnknowns(), PETSC_DETERMINE, PETSC_DETERMINE, this, &pMat);
  MatShellSetOperation(pMat, MATOP_MULT, (void(*)(void))matMultPIMS);

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

PetscErrorCode preconPIMS(PC pc, Vec x, Vec y) {
  PETScInvMassSolver *solver;
  PCShellGetContext(pc, (void **)&solver);
  solver->precond(x, y);
  return 0;
}

void PETScInvMassSolver::set_shell_pc(PC pc) {
  PCShellSetApply(pc, preconPIMS);
  PCShellSetContext(pc, this);
}

void PETScInvMassSolver::set_tol_and_iter(const double rtol, const double atol, const int maxiter) {
  KSPSetTolerances(ksp, rtol, atol, 1e5, maxiter);
}