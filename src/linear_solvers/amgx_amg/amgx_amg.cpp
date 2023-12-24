#include "dg_linear_solvers/amgx_amg.h"

#include "dg_matrices/poisson_coarse_matrix.h"
#include "timing.h"
#include "config.h"
#include "op2_utils.h"
#include "dg_abort.h"

#include <iostream>
#include <type_traits>
#include <cuda_runtime_api.h>

extern Timing *timer;
extern Config *config;

AMGX_resources_handle amgx_res_handle;
AMGX_config_handle amgx_config_handle;

void print_callback(const char *msg, int length) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) { printf("%s", msg); }
}

AmgXAMGSolver::AmgXAMGSolver(DGMesh *m) {
  bc = nullptr;
  mesh = m;
  nullspace = false;

  std::string amgx_config_path = "";
  if(!config->getStr("amgx", "amgx_config_file", amgx_config_path))
    dg_abort("AmgX configuration file not specified ('amgx_config_file' in ini file)");

  AMGX_SAFE_CALL(AMGX_initialize());
  AMGX_SAFE_CALL(AMGX_initialize_plugins());
  AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
  AMGX_SAFE_CALL(AMGX_install_signal_handler());
  AMGX_SAFE_CALL(AMGX_config_create_from_file(&amgx_config_handle, amgx_config_path.c_str()));

  #ifdef DG_MPI
  amgx_comm = MPI_COMM_WORLD;
  int gpu_num;
  cudaGetDevice(&gpu_num);
  int devices[] = {gpu_num};
  AMGX_resources_create(&amgx_res_handle, amgx_config_handle, &amgx_comm, 1, devices);
  #else
  AMGX_resources_create_simple(&amgx_res_handle, amgx_config_handle);
  #endif

  AMGX_vector_create(&rhs_amgx, amgx_res_handle, AMGX_mode_dFFI);
  AMGX_vector_create(&soln_amgx, amgx_res_handle, AMGX_mode_dFFI);
  AMGX_solver_create(&solver_amgx, amgx_res_handle, AMGX_mode_dFFI, amgx_config_handle);
}

AmgXAMGSolver::~AmgXAMGSolver() {
  // TODO destroy matrix
  AMGX_matrix_destroy(*amgx_mat);
  AMGX_solver_destroy(solver_amgx);
  AMGX_vector_destroy(rhs_amgx);
  AMGX_vector_destroy(soln_amgx);
  AMGX_config_destroy(amgx_config_handle);
  AMGX_resources_destroy(amgx_res_handle);
  AMGX_finalize_plugins();
  AMGX_finalize();
}

bool AmgXAMGSolver::solve(op_dat rhs, op_dat ans) {
  timer->startTimer("AmgXAMGSolver - solve");
  if(dynamic_cast<PoissonCoarseMatrix*>(matrix) == nullptr) {
    dg_abort("AmgXAMGSolver matrix should be of type PoissonCoarseMatrix\n");
  }

  PoissonCoarseMatrix *coarse_mat = dynamic_cast<PoissonCoarseMatrix*>(matrix);

  timer->startTimer("AmgXAMGSolver - Get matrix and setup");
  if(coarse_mat->getAmgXMat(&amgx_mat)) {
    AMGX_vector_bind(rhs_amgx, *amgx_mat);
    AMGX_vector_bind(soln_amgx, *amgx_mat);
    AMGX_solver_setup(solver_amgx, *amgx_mat);
  }
  timer->endTimer("AmgXAMGSolver - Get matrix and setup");

  timer->startTimer("AmgXAMGSolver - Transfer vec");
  float *rhs_ptr = getOP2PtrHostSP(rhs, OP_READ);
  float *ans_ptr = getOP2PtrHostSP(ans, OP_READ);

  float *rhs_amgx_ptr = (float *)malloc(coarse_mat->getUnknowns() * sizeof(float));
  float *ans_amgx_ptr = (float *)malloc(coarse_mat->getUnknowns() * sizeof(float));

  for(int i = 0; i < mesh->cells->size; i++) {
    for(int j = 0; j < DG_NP_N1; j++) {
      rhs_amgx_ptr[i * DG_NP_N1 + j] = rhs_ptr[i * DG_NP + j];
      ans_amgx_ptr[i * DG_NP_N1 + j] = ans_ptr[i * DG_NP + j];
    }
  }

  releaseOP2PtrHostSP(rhs, OP_READ, rhs_ptr);
  releaseOP2PtrHostSP(ans, OP_READ, ans_ptr);

  AMGX_SAFE_CALL(AMGX_pin_memory(rhs_amgx_ptr, coarse_mat->getUnknowns() * sizeof(float)));
  AMGX_SAFE_CALL(AMGX_pin_memory(ans_amgx_ptr, coarse_mat->getUnknowns() * sizeof(float)));

  AMGX_vector_upload(rhs_amgx,  coarse_mat->getUnknowns(), 1, rhs_amgx_ptr);
  AMGX_vector_upload(soln_amgx, coarse_mat->getUnknowns(), 1, ans_amgx_ptr);

  AMGX_SAFE_CALL(AMGX_unpin_memory(rhs_amgx_ptr));
  AMGX_SAFE_CALL(AMGX_unpin_memory(ans_amgx_ptr));

  free(rhs_amgx_ptr);
  free(ans_amgx_ptr);
  timer->endTimer("AmgXAMGSolver - Transfer vec");

  timer->startTimer("AmgXAMGSolver - AmgX Solve Call");
  AMGX_solver_solve(solver_amgx, rhs_amgx, soln_amgx);
  timer->endTimer("AmgXAMGSolver - AmgX Solve Call");

  AMGX_SOLVE_STATUS status;
  AMGX_solver_get_status(solver_amgx, &status);
  if(status != AMGX_SOLVE_SUCCESS) {
    int iter_num;
    AMGX_solver_get_iterations_number(solver_amgx, &iter_num);
    switch(status) {
      case AMGX_SOLVE_FAILED:
        dg_abort("AMGX solve failed after " + std::to_string(iter_num) + " iterations");
      case AMGX_SOLVE_DIVERGED:
        dg_abort("AMGX solve diverged after " + std::to_string(iter_num) + " iterations");
      default:
        dg_abort("AMGX solve failed, unrecognised status");
    }
  }

  timer->startTimer("AmgXAMGSolver - Transfer vec");
  float *ans_ptr_w = getOP2PtrHostSP(ans, OP_WRITE);
  float *ans_amgx_ptr_w = (float *)malloc(coarse_mat->getUnknowns() * sizeof(float));

  AMGX_vector_download(soln_amgx, ans_amgx_ptr_w);

  for(int i = 0; i < mesh->cells->size; i++) {
    for(int j = 0; j < DG_NP_N1; j++) {
      ans_ptr_w[i * DG_NP + j] = ans_amgx_ptr_w[i * DG_NP_N1 + j];
    }
  }

  free(ans_amgx_ptr_w);
  releaseOP2PtrHostSP(ans, OP_WRITE, ans_ptr_w);
  timer->endTimer("AmgXAMGSolver - Transfer vec");

  timer->endTimer("AmgXAMGSolver - solve");

  return true;
}
