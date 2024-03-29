#include "dg_linear_solvers/petsc_block_jacobi.h"

#include "op_seq.h"

#include "op2_utils.h"
#include "dg_linear_solvers/petsc_utils.h"
#include "timing.h"
#include "config.h"
#include "dg_dat_pool.h"

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <type_traits>
#include <stdexcept>

extern Timing *timer;
extern DGDatPool *dg_dat_pool;
extern Config *config;

PETScBlockJacobiSolver::PETScBlockJacobiSolver(DGMesh *m) {
  bc = nullptr;
  nullspace = false;
  pMatInit = false;
  mesh = m;

  pre = op_decl_dat(mesh->cells, DG_NP * DG_NP, DG_FP_STR, (DG_FP *)NULL, "block_jacobi_pre");

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

PETScBlockJacobiSolver::~PETScBlockJacobiSolver() {
  MatDestroy(&pMat);
  KSPDestroy(&ksp);
}

void PETScBlockJacobiSolver::set_matrix(PoissonMatrix *mat) {
  if(dynamic_cast<PoissonMatrixFreeBlockDiag*>(mat) == nullptr)
    throw std::runtime_error("PETScBlockJacobiSolver matrix should be of type PoissonMatrixFreeBlockDiag\n");
  
  matrix = mat;
  block_matrix = dynamic_cast<PoissonMatrixFreeBlockDiag*>(mat);
}

bool PETScBlockJacobiSolver::solve(op_dat rhs, op_dat ans) {
  timer->startTimer("PETScBlockJacobiSolver - solve");
  // TODO only call when necessary
  calc_precond_mat();
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
    block_matrix->apply_bc(rhs, bc);

  Vec b, x;
  PETScUtils::create_vec(&b, mesh->cells);
  PETScUtils::create_vec(&x, mesh->cells);

  PETScUtils::load_vec(&b, rhs);
  PETScUtils::load_vec(&x, ans);

  timer->startTimer("PETScBlockJacobiSolver - KSPSolve");
  KSPSolve(ksp, b, x);
  timer->endTimer("PETScBlockJacobiSolver - KSPSolve");

  int numIt;
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

  timer->endTimer("PETScBlockJacobiSolver - solve");

  return converged;
}

void PETScBlockJacobiSolver::calc_rhs(Vec in, Vec out) {
  timer->startTimer("PETScBlockJacobiSolver - calc_rhs");
  // Copy u to OP2 dat
  DGTempDat tmp_in  = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_out = dg_dat_pool->requestTempDatCells(DG_NP);
  PETScUtils::store_vec(&in, tmp_in.dat);

  block_matrix->mult(tmp_in.dat, tmp_out.dat);

  PETScUtils::load_vec(&out, tmp_out.dat);
  dg_dat_pool->releaseTempDatCells(tmp_in);
  dg_dat_pool->releaseTempDatCells(tmp_out);
  timer->endTimer("PETScBlockJacobiSolver - calc_rhs");
}

// Matrix-free block-jacobi preconditioning function
void PETScBlockJacobiSolver::precond(Vec in, Vec out) {
  timer->startTimer("PETScBlockJacobiSolver - precond");
  DGTempDat tmp_in  = dg_dat_pool->requestTempDatCells(DG_NP);
  DGTempDat tmp_out = dg_dat_pool->requestTempDatCells(DG_NP);
  PETScUtils::store_vec(&in, tmp_in.dat);

  op_par_loop(block_jacobi_pre, "block_jacobi_pre", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(tmp_in.dat,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(pre, -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(tmp_out.dat, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  PETScUtils::load_vec(&out, tmp_out.dat);
  dg_dat_pool->releaseTempDatCells(tmp_in);
  dg_dat_pool->releaseTempDatCells(tmp_out);
  timer->endTimer("PETScBlockJacobiSolver - precond");
}

void PETScBlockJacobiSolver::calc_precond_mat() {
  timer->startTimer("PETScBlockJacobiSolver - calc_precond_mat");
  const DG_FP *block_diag_ptr = getOP2PtrHost(block_matrix->block_diag, OP_READ);
  DG_FP *pre_ptr = getOP2PtrHost(pre, OP_WRITE);

  #pragma omp parallel for
  for(int i = 0; i < mesh->cells->size; i++) {
    const DG_FP *in_c = block_diag_ptr + i * block_matrix->block_diag->dim;
    DG_FP *inv_c      = pre_ptr + i * pre->dim;

    arma::Mat<DG_FP> a(in_c, DG_NP, DG_NP);
    arma::Mat<DG_FP> b(inv_c, DG_NP, DG_NP, false, true);

    #ifdef DG_COL_MAJ
    b = arma::inv(a);
    #else
    b = arma::inv(a.t()).t();
    #endif
    // b = arma::inv_sympd(a);
  }

  releaseOP2PtrHost(block_matrix->block_diag, OP_READ, block_diag_ptr);
  releaseOP2PtrHost(pre, OP_WRITE, pre_ptr);
  timer->endTimer("PETScBlockJacobiSolver - calc_precond_mat");
}

PetscErrorCode matMultPBJS(Mat A, Vec x, Vec y) {
  PETScBlockJacobiSolver *solver;
  MatShellGetContext(A, &solver);
  solver->calc_rhs(x, y);
  return 0;
}

void PETScBlockJacobiSolver::create_shell_mat() {
  if(pMatInit)
    MatDestroy(&pMat);

  MatCreateShell(PETSC_COMM_WORLD, block_matrix->getUnknowns(), block_matrix->getUnknowns(), PETSC_DETERMINE, PETSC_DETERMINE, this, &pMat);
  MatShellSetOperation(pMat, MATOP_MULT, (void(*)(void))matMultPBJS);

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

PetscErrorCode preconPBJS(PC pc, Vec x, Vec y) {
  PETScBlockJacobiSolver *solver;
  PCShellGetContext(pc, (void **)&solver);
  solver->precond(x, y);
  return 0;
}

void PETScBlockJacobiSolver::set_shell_pc(PC pc) {
  PCShellSetApply(pc, preconPBJS);
  PCShellSetContext(pc, this);
}

void PETScBlockJacobiSolver::set_tol_and_iter(const double rtol, const double atol, const int maxiter) {
  KSPSetTolerances(ksp, rtol, atol, 1e5, maxiter);
}