#include "dg_linear_solvers/pmultigrid.h"

#include "op_seq.h"

#include <random>
#include <string>
#include <stdexcept>

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "dg_matrices/3d/poisson_semi_matrix_free_3d.h"
#include "op2_utils.h"
#include "timing.h"
#include "config.h"
#include "dg_dat_pool.h"

#include "dg_linear_solvers/amgx_amg.h"
#include "dg_linear_solvers/hypre_amg.h"
#include "dg_linear_solvers/petsc_amg_coarse.h"

extern Timing *timer;
extern Config *config;
extern DGDatPool *dg_dat_pool;

#define RAND_VEC_SIZE 25
#define MAX_ITER_EIG_APPROX 10

std::vector<int> parseInts(const std::string &str) {
  std::vector<int> result;
  std::stringstream ss(str);
  while(ss.good()) {
    std::string sub_str;
    std::getline(ss, sub_str, ',');
    if(sub_str != " ")
      result.push_back(std::stoi(sub_str));
  }
  return result;
}

PMultigridPoissonSolver::PMultigridPoissonSolver(DGMesh *m) {
  bc = nullptr;
  mesh = m;

  std::string orders_str;
  if(config->getStr("p-multigrid", "orders", orders_str)) {
    orders = parseInts(orders_str);
    bool contains_first_order = false;
    for(const int &o : orders) {
      if(o == 1)
        contains_first_order = true;
    }
    if(!contains_first_order) {
      throw std::runtime_error("\nParsed orders for P-Multigrid does not contain a first order solve.\n");
    }
  } else {
    int tmp_order = DG_ORDER;
    while(tmp_order != 1) {
      orders.push_back(tmp_order);
      tmp_order /= 2;
    }
  }

  num_levels = orders.size();

  std::string pre_str;
  if(config->getStr("p-multigrid", "pre_it", pre_str)) {
    pre_it = parseInts(pre_str);
    if(!(pre_it.size() == num_levels)) {
      throw std::runtime_error("\nParsed pre smoothing iterations for P-Multigrid does not match number of levels.\n");
    }
  } else {
    pre_it.push_back(20);
    for(int i = 1; i < num_levels; i++) {
      pre_it.push_back(pre_it[i - 1] / 4);
    }
  }

  std::string post_str;
  if(config->getStr("p-multigrid", "post_it", post_str)) {
    post_it = parseInts(post_str);
    if(!(post_it.size() == num_levels)) {
      throw std::runtime_error("\nParsed post smoothing iterations for P-Multigrid does not match number of levels.\n");
    }
  } else {
    post_it.push_back(10);
    for(int i = 1; i < num_levels; i++) {
      post_it.push_back(post_it[i - 1] / 4);
    }
  }

  std::string cheb_str;
  if(config->getStr("p-multigrid", "cheb_orders", cheb_str)) {
    cheb_orders = parseInts(cheb_str);
    if(!(cheb_orders.size() == num_levels)) {
      throw std::runtime_error("\nParsed Chebyshev orders for P-Multigrid does not match number of levels.\n");
    }
  } else {
    for(int i = 0; i < num_levels; i++) {
      cheb_orders.push_back(2);
    }
  }

  eigen_val_saftey_factor = 1.1;
  config->getDouble("p-multigrid", "eigen_val_saftey_factor", eigen_val_saftey_factor);

  std::string smoother_str;
  smoother = CHEBYSHEV;
  if(config->getStr("p-multigrid", "smoother", smoother_str) && smoother_str == "jacobi") {
      smoother = JACOBI;
  }

  float *tmp_data = (float *)calloc(DG_NP * mesh->cells->size, sizeof(float));
  std::string name;
  for(int i = 0; i < num_levels; i++) {
    name = "p_multigrid_diag" + std::to_string(i);
    diag_dats.push_back(op_decl_dat(mesh->cells, DG_NP, "float", tmp_data, name.c_str()));
  }

  free(tmp_data);

  std::string coarseSolver_str;
  coarseSolver_type = PETSC;
  if(config->getStr("p-multigrid", "coarse_solver", coarseSolver_str)) {
      if(coarseSolver_str == "petsc") {
        coarseSolver_type = PETSC;
      } else if(coarseSolver_str == "amgx") {
        coarseSolver_type = AMGX;
      } else if(coarseSolver_str == "hypre") {
        coarseSolver_type = HYPRE;
      } else {
        op_printf("Unrecognised coarse solver for p-multigrid, defaulting to PETSc\n");
      }
  }

  switch(coarseSolver_type) {
    case PETSC:
      coarseSolver = new PETScAMGCoarseSolver(mesh);
      break;
    case AMGX:
      #if defined(INS_BUILD_WITH_AMGX) && defined(OP2_DG_CUDA)
      coarseSolver = new AmgXAMGSolver(mesh);
      #else
      throw std::runtime_error("Not built with AmgX");
      #endif
      break;
    case HYPRE:
      #ifdef INS_BUILD_WITH_HYPRE
      coarseSolver = new HYPREAMGSolver(mesh);
      #else
      throw std::runtime_error("Not built with HYPRE");
      #endif
      break;
  }
}

PMultigridPoissonSolver::~PMultigridPoissonSolver() {
  delete coarseSolver;
}

void PMultigridPoissonSolver::init() {
  coarseSolver->init();
}

void PMultigridPoissonSolver::set_matrix(PoissonMatrix *mat) {
  if(dynamic_cast<PoissonSemiMatrixFree*>(mat) == nullptr && dynamic_cast<PoissonMatrixFreeDiag*>(mat) == nullptr) {
    throw std::runtime_error("PMultigridPoissonSolver matrix should be of type PoissonSemiMatrixFree or PoissonMatrixFreeDiag\n");
  }
  matrix = mat;
  if(dynamic_cast<PoissonSemiMatrixFree*>(mat)) {
    smfMatrix = dynamic_cast<PoissonSemiMatrixFree*>(mat);
    diagMat = false;
  } else {
    mfdMatrix = dynamic_cast<PoissonMatrixFreeDiag*>(mat);
    diagMat = true;
  }

  eig_vals.clear();
  int current_mesh_order = mesh->order_int;
  std::vector<op_dat> empty_vec;
  timer->startTimer("PMultigridPoissonSolver - Calc Eigen Values");
  for(int i = 0; i < orders.size(); i++) {
    mesh->update_order(orders[i], empty_vec);
    if(diagMat) {
      mfdMatrix->calc_mat_partial();
      op_par_loop(copy_dg_np_dp2sp_tk, "copy_dg_np_dp2sp_tk", mesh->cells,
                  op_arg_dat(mfdMatrix->diag, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
                  op_arg_dat(diag_dats[i], -1, OP_ID, DG_NP, "float", OP_WRITE));
    } else {
      smfMatrix->calc_mat_partial();
      op_par_loop(copy_diag, "copy_diag", mesh->cells,
                  op_arg_dat(mesh->order,    -1, OP_ID, 1, "int", OP_READ),
                  op_arg_dat(smfMatrix->op1, -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
                  op_arg_dat(diag_dats[i],   -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));
    }
    eig_vals.push_back(maxEigenValue());
  }
  timer->endTimer("PMultigridPoissonSolver - Calc Eigen Values");
  mesh->update_order(current_mesh_order, empty_vec);
}

bool PMultigridPoissonSolver::solve(op_dat rhs, op_dat ans) {
  timer->startTimer("PMultigridPoissonSolver - solve");
  // if(bc)
  //   matrix->apply_bc(rhs, bc);

  int order = DG_ORDER;

  setupDirectSolve();

  // Get temporary dats from pool
  std::vector<DGTempDat> tmp_dats;
  u_dat.clear();
  b_dat.clear();
  for(int i = 0; i < num_levels; i++) {
    DGTempDat tmp0 = dg_dat_pool->requestTempDatCellsSP(DG_NP);
    DGTempDat tmp1 = dg_dat_pool->requestTempDatCellsSP(DG_NP);
    tmp_dats.push_back(tmp0);
    tmp_dats.push_back(tmp1);
    u_dat.push_back(tmp0.dat);
    b_dat.push_back(tmp1.dat);
  }

  op_par_loop(copy_dg_np_dp2sp_tk, "copy_dg_np_dp2sp_tk", mesh->cells,
              op_arg_dat(ans, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(u_dat[0], -1, OP_ID, DG_NP, "float", OP_WRITE));
  op_par_loop(copy_dg_np_dp2sp_tk, "copy_dg_np_dp2sp_tk", mesh->cells,
              op_arg_dat(rhs, -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(b_dat[0], -1, OP_ID, DG_NP, "float", OP_WRITE));

  cycle(order, 0);

  op_par_loop(copy_dg_np_sp2dp_tk, "copy_dg_np_sp2dp_tk", mesh->cells,
              op_arg_dat(u_dat[0], -1, OP_ID, DG_NP, "float", OP_READ),
              op_arg_dat(ans, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));

  for(int i = 0; i < tmp_dats.size(); i++) {
    dg_dat_pool->releaseTempDatCellsSP(tmp_dats[i]);
  }
  timer->endTimer("PMultigridPoissonSolver - solve");
  return true;
}

void PMultigridPoissonSolver::cycle(int order, const int level) {
  // Relaxation
  // u = u + R^-1 (F - Au)
  timer->startTimer("PMultigridPoissonSolver - Relaxation");
  smooth(pre_it[level], level);
  timer->endTimer("PMultigridPoissonSolver - Relaxation");

  if(order == 1) {
    // u = A^-1 (F)
    if(coarseMatCalcRequired) {
      timer->startTimer("PMultigridPoissonSolver - Calc Mat");
      coarseMatrix->calc_mat();
      coarseMatCalcRequired = false;
      timer->endTimer("PMultigridPoissonSolver - Calc Mat");
    }

    timer->startTimer("PMultigridPoissonSolver - Direct Solve");
    coarseSolver->solve(b_dat[level], u_dat[level]);
    timer->endTimer("PMultigridPoissonSolver - Direct Solve");

    // Relaxation
    // u = u + R^-1 (F - Au)
    timer->startTimer("PMultigridPoissonSolver - Relaxation");
    smooth(post_it[level], level);
    timer->endTimer("PMultigridPoissonSolver - Relaxation");
    return;
  }

  // Restriction
  timer->startTimer("PMultigridPoissonSolver - Restriction");
  int order_new = orders[level + 1];
  // F = I^T (F - Au)
  DGTempDat tmp_dat = dg_dat_pool->requestTempDatCellsSP(DG_NP);
  matrix->mult_sp(u_dat[level], tmp_dat.dat);
  op_par_loop(p_multigrid_restriction, "p_multigrid_restriction", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(tmp_dat.dat,    -1, OP_ID, DG_NP, "float", OP_READ),
              op_arg_dat(b_dat[level],   -1, OP_ID, DG_NP, "float", OP_READ),
              op_arg_dat(b_dat[level+1], -1, OP_ID, DG_NP, "float", OP_WRITE),
              op_arg_dat(u_dat[level+1], -1, OP_ID, DG_NP, "float", OP_WRITE));
  dg_dat_pool->releaseTempDatCellsSP(tmp_dat);

  std::vector<op_dat> dats_to_update;
  dats_to_update.push_back(b_dat[level+1]);
  timer->startTimer("PMultigridPoissonSolver - Interp");
  mesh->update_order_sp(order_new, dats_to_update);
  timer->endTimer("PMultigridPoissonSolver - Interp");
  timer->endTimer("PMultigridPoissonSolver - Restriction");

  cycle(order_new, level + 1);

  // Prologation
  // u = u + Iu
  timer->startTimer("PMultigridPoissonSolver - Prolongation");
  std::vector<op_dat> dats_to_update2;
  dats_to_update2.push_back(u_dat[level+1]);
  timer->startTimer("PMultigridPoissonSolver - Interp");
  mesh->update_order_sp(order, dats_to_update2);
  timer->endTimer("PMultigridPoissonSolver - Interp");

  op_par_loop(p_multigrid_prolongation, "p_multigrid_prolongation", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(u_dat[level+1], -1, OP_ID, DG_NP, "float", OP_READ),
              op_arg_dat(u_dat[level],   -1, OP_ID, DG_NP, "float", OP_RW));
  timer->endTimer("PMultigridPoissonSolver - Prolongation");

  // Relaxation
  // u = u + R^-1 (F - Au)
  timer->startTimer("PMultigridPoissonSolver - Relaxation");
  smooth(post_it[level], level);
  timer->endTimer("PMultigridPoissonSolver - Relaxation");
}

void PMultigridPoissonSolver::set_coarse_matrix(PoissonCoarseMatrix *c_mat) {
  coarseMatrix = c_mat;
  coarseSolver->set_matrix(coarseMatrix);
  coarseMatCalcRequired = true;
}

void PMultigridPoissonSolver::setupDirectSolve() {
  // coarseSolver->set_bcs(bc);
  coarseSolver->set_nullspace(nullspace);
}

DG_FP PMultigridPoissonSolver::maxEigenValue() {
  // Get approx eigenvector using power iteration
  const int N = mesh->order_int;
  const int k = std::min(MAX_ITER_EIG_APPROX, (N + 1) * (N + 2) * (N + 3) / 6);
  std::vector<DGTempDat> tmp_dats;
  eigen_tmps.clear();
  for(int i = 0; i < k; i++) {
    DGTempDat tmp0 = dg_dat_pool->requestTempDatCellsSP(DG_NP);
    tmp_dats.push_back(tmp0);
    eigen_tmps.push_back(tmp0.dat);
  }

  DGTempDat tmp_eg = dg_dat_pool->requestTempDatCellsSP(DG_NP);

  timer->startTimer("PMultigridPoissonSolver - Random Vec");
  setRandomVector(tmp_eg.dat);
  timer->endTimer("PMultigridPoissonSolver - Random Vec");

  float norm = 0.0;
  op_par_loop(p_multigrid_vec_norm, "p_multigrid_vec_norm", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(tmp_eg.dat, -1, OP_ID, DG_NP, "float", OP_READ),
              op_arg_gbl(&norm, 1, "float", OP_INC));

  norm = sqrt(norm);
  op_par_loop(p_multigrid_vec_normalise, "p_multigrid_vec_normalise", mesh->cells,
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(tmp_eg.dat, -1, OP_ID, DG_NP, "float", OP_READ),
              op_arg_gbl(&norm, 1, "float", OP_READ),
              op_arg_dat(eigen_tmps[0], -1, OP_ID, DG_NP, "float", OP_WRITE));

  arma::mat H(k, k, arma::fill::zeros);
  for(int n = 0; n < k; n++) {
    matrix->multJacobi_sp(eigen_tmps[n], tmp_eg.dat);

    for(int j = 0; j <= n; j++) {
      float dot = 0.0;
      op_par_loop(p_multigrid_vec_dot, "p_multigrid_vec_dot", mesh->cells,
                  op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                  op_arg_dat(eigen_tmps[j], -1, OP_ID, DG_NP, "float", OP_READ),
                  op_arg_dat(tmp_eg.dat, -1, OP_ID, DG_NP, "float", OP_READ),
                  op_arg_gbl(&dot, 1, "float", OP_INC));
      H(j,n) = dot;

      op_par_loop(p_multigrid_vec_minus, "p_multigrid_vec_minus", mesh->cells,
                  op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                  op_arg_dat(eigen_tmps[j], -1, OP_ID, DG_NP, "float", OP_READ),
                  op_arg_dat(tmp_eg.dat, -1, OP_ID, DG_NP, "float", OP_RW),
                  op_arg_gbl(&dot, 1, "float", OP_READ));
    }

    if(n + 1 < k) {
      float norm = 0.0;
      op_par_loop(p_multigrid_vec_norm, "p_multigrid_vec_norm", mesh->cells,
                  op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                  op_arg_dat(tmp_eg.dat, -1, OP_ID, DG_NP, "float", OP_READ),
                  op_arg_gbl(&norm, 1, "float", OP_INC));

      norm = sqrt(norm);
      op_par_loop(p_multigrid_vec_normalise, "p_multigrid_vec_normalise", mesh->cells,
                  op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                  op_arg_dat(tmp_eg.dat, -1, OP_ID, DG_NP, "float", OP_READ),
                  op_arg_gbl(&norm, 1, "float", OP_READ),
                  op_arg_dat(eigen_tmps[n+1], -1, OP_ID, DG_NP, "float", OP_WRITE));
      H(n+1,n) = norm;
    }
  }

  for(int i = 0; i < tmp_dats.size(); i++) {
    dg_dat_pool->releaseTempDatCellsSP(tmp_dats[i]);
  }
  dg_dat_pool->releaseTempDatCellsSP(tmp_eg);

  auto eigen_values = arma::eig_gen(H);
  // std::cout << eigen_values << std::endl;
  double max_ = 0.0;
  for(int i = 0; i < eigen_values.n_elem; i++) {
    double tmp = std::abs(eigen_values[i]);
    max_ = std::max(max_, tmp);
  }

  return eigen_val_saftey_factor * max_;
}

void PMultigridPoissonSolver::setRandomVector(op_dat vec) {
  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  float rand_vec[RAND_VEC_SIZE];
  for(int i = 0; i < RAND_VEC_SIZE; i++) {
    rand_vec[i] = dist(eng);
  }

  float *vec_ptr = getOP2PtrHostSP(vec, OP_WRITE);

  #pragma omp parallel for
  for(int i = 0; i < vec->set->size * vec->dim; i++) {
    vec_ptr[i] = rand_vec[i % RAND_VEC_SIZE];
  }

  releaseOP2PtrHostSP(vec, OP_WRITE, vec_ptr);
}

void PMultigridPoissonSolver::smooth(const int iter, const int level) {
  max_eig = eig_vals[level];
  w = (4.0 / 3.0) * (1.0 / max_eig);
  switch(smoother) {
    case JACOBI:
      for(int i = 0; i < iter; i++) {
        jacobi_smoother(level);
      }
      break;
    case CHEBYSHEV:
      for(int i = 0; i < iter; i++) {
        chebyshev_smoother(level);
      }
      break;
  }
}

void PMultigridPoissonSolver::jacobi_smoother(const int level) {
  throw std::runtime_error("jacobi_smoother not implemented\n");
  /*
  DGTempDat tmp_dat = dg_dat_pool->requestTempDatCells(DG_NP);
  matrix->mult(u_dat[level], tmp_dat.dat);
  op_par_loop(p_multigrid_relaxation_jacobi_diag, "p_multigrid_relaxation_jacobi_diag", mesh->cells,
              op_arg_gbl(&w, 1, DG_FP_STR, OP_READ),
              op_arg_dat(mesh->order,      -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(tmp_dat.dat,      -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(b_dat[level],     -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(diag_dats[level], -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(u_dat[level],     -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  dg_dat_pool->releaseTempDatCells(tmp_dat);
  */
}

void PMultigridPoissonSolver::chebyshev_smoother(const int level) {
  const float lamda1 = max_eig;
  const float lamda0 = lamda1 / 10.0;
  const float theta = 0.5 * (lamda1 + lamda0);
  const float delta = 0.5 * (lamda1 - lamda0);
  const float invTheta = 1.0 / theta;
  const float sigma = theta / delta;
  float rho_n = 1.0 / sigma;
  float rho_np1;
  DGTempDat tmp0 = dg_dat_pool->requestTempDatCellsSP(DG_NP);
  DGTempDat tmp1 = dg_dat_pool->requestTempDatCellsSP(DG_NP);
  DGTempDat tmp2 = dg_dat_pool->requestTempDatCellsSP(DG_NP);
  op_dat RES = tmp0.dat;
  op_dat Ad  = tmp1.dat;
  op_dat d   = tmp2.dat;

  timer->startTimer("PMultigridPoissonSolver - Relaxation - Mult");
  matrix->mult_sp(u_dat[level], RES);
  timer->endTimer("PMultigridPoissonSolver - Relaxation - Mult");

  op_par_loop(p_multigrid_relaxation_chebyshev_0, "p_multigrid_relaxation_chebyshev_0", mesh->cells,
              op_arg_gbl(&invTheta, 1, "float", OP_READ),
              op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(b_dat[level],     -1, OP_ID, DG_NP, "float", OP_READ),
              op_arg_dat(diag_dats[level], -1, OP_ID, DG_NP, "float", OP_READ),
              op_arg_dat(RES, -1, OP_ID, DG_NP, "float", OP_RW),
              op_arg_dat(d,   -1, OP_ID, DG_NP, "float", OP_WRITE),
              op_arg_dat(u_dat[level], -1, OP_ID, DG_NP, "float", OP_RW));

  for(int i = 0; i < cheb_orders[level]; i++) {
    timer->startTimer("PMultigridPoissonSolver - Relaxation - Mult");
    matrix->mult_sp(d, Ad);
    timer->endTimer("PMultigridPoissonSolver - Relaxation - Mult");

    rho_np1 = 1.0 / (2.0 * sigma - rho_n);
    float rhoDivDelta = 2.0 * rho_np1 / delta;
    float tmp = rho_np1 * rho_n;

    op_par_loop(p_multigrid_relaxation_chebyshev_2, "p_multigrid_relaxation_chebyshev_2", mesh->cells,
                op_arg_gbl(&mesh->order_int, 1, "int", OP_READ),
                op_arg_gbl(&rhoDivDelta, 1, "float", OP_READ),
                op_arg_gbl(&tmp, 1, "float", OP_READ),
                op_arg_dat(Ad, -1, OP_ID, DG_NP, "float", OP_READ),
                op_arg_dat(diag_dats[level], -1, OP_ID, DG_NP, "float", OP_READ),
                op_arg_dat(RES, -1, OP_ID, DG_NP, "float", OP_RW),
                op_arg_dat(d, -1, OP_ID, DG_NP, "float", OP_RW),
                op_arg_dat(u_dat[level], -1, OP_ID, DG_NP, "float", OP_RW));

    rho_n = rho_np1;
  }

  dg_dat_pool->releaseTempDatCellsSP(tmp0);
  dg_dat_pool->releaseTempDatCellsSP(tmp1);
  dg_dat_pool->releaseTempDatCellsSP(tmp2);
}
