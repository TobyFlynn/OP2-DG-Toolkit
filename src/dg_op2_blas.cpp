#include "dg_op2_blas.h"

#include "op_seq.h"

#include <iostream>

#if DG_DIM == 2
#include "2d/dg_global_constants.h"
#elif DG_DIM == 3
#include "3d/dg_global_constants.h"
#endif

extern DGConstants *constants;

void op2_gemv_inv_mass_gass_interpT(DGMesh *mesh, bool transpose,
                                    const double alpha, op_dat x,
                                    const double beta, op_dat y) {
  if(transpose) {
    std::cerr << "op2_gemv_inv_mass_gass_interpT not implemented for transpose ... exiting" << std::endl;
  } else {
    op_par_loop(gemv_inv_mass_gauss_interpT, "gemv_inv_mass_gauss_interpT", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::INV_MASS_GAUSS_INTERP_T), DG_ORDER * DG_G_NP * DG_NP, "double", OP_READ),
                op_arg_dat(mesh->J, -1, OP_ID, DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_G_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
  }
}

void op2_gemv_gauss_interp(DGMesh *mesh, bool transpose, const double alpha,
                           op_dat x, const double beta, op_dat y) {
  if(transpose) {
    op_par_loop(gemv_gauss_interpT, "gemv_gauss_interpT", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_INTERP), DG_ORDER * DG_G_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_G_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
  } else {
    op_par_loop(gemv_gauss_interp, "gemv_gauss_interp", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::GAUSS_INTERP), DG_ORDER * DG_G_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_G_NP, "double", OP_RW));
  }
}

void op2_gemv_lift(DGMesh *mesh, bool transpose, const double alpha, op_dat x,
                   const double beta, op_dat y) {
  if(transpose) {
    op_par_loop(gemv_liftT, "gemv_liftT", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::LIFT), DG_ORDER * 3 * DG_NPF * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, 3 * DG_NPF, "double", OP_RW));
  } else {
    op_par_loop(gemv_lift, "gemv_lift", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(constants->get_mat_ptr(DGConstants::LIFT), DG_ORDER * 3 * DG_NPF * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, 3 * DG_NPF, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
  }
}

void op2_gemv_np_np(DGMesh *mesh, bool transpose, const double alpha, 
                    const double *matrix, op_dat x, const double beta, 
                    op_dat y) {
  if(transpose) {
    op_par_loop(gemv_np_npT, "gemv_np_npT", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(matrix, DG_ORDER * DG_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
  } else {
    op_par_loop(gemv_np_np, "gemv_np_np", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(matrix, DG_ORDER * DG_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
  }
}

void op2_gemv_cub_np_np(DGMesh *mesh, bool transpose, const double alpha, 
                        const double *matrix, op_dat x, const double beta, 
                        op_dat y) {
  if(transpose) {
    op_par_loop(gemv_cub_np_npT, "gemv_cub_np_npT", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(matrix, DG_ORDER * DG_CUB_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
  } else {
    op_par_loop(gemv_cub_np_np, "gemv_cub_np_np", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(matrix, DG_ORDER * DG_CUB_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_CUB_NP, "double", OP_RW));
  }
}

void op2_gemv(DGMesh *mesh, bool transpose, const double alpha,
              DGConstants::Constant_Matrix matrix, op_dat x, const double beta,
              op_dat y) {
  switch(matrix) {
    case DGConstants::INV_MASS_GAUSS_INTERP_T:
      op2_gemv_inv_mass_gass_interpT(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::GAUSS_INTERP:
      op2_gemv_gauss_interp(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::CUB_DR:
      op2_gemv_cub_np_np(mesh, transpose, alpha, constants->get_mat_ptr(matrix), x, beta, y);
      break;
    case DGConstants::CUB_DS:
      op2_gemv_cub_np_np(mesh, transpose, alpha, constants->get_mat_ptr(matrix), x, beta, y);
      break;
    case DGConstants::DR:
      op2_gemv_np_np(mesh, transpose, alpha, constants->get_mat_ptr(matrix), x, beta, y);
      break;
    case DGConstants::DS:
      op2_gemv_np_np(mesh, transpose, alpha, constants->get_mat_ptr(matrix), x, beta, y);
      break;
    case DGConstants::DRW:
      op2_gemv_np_np(mesh, transpose, alpha, constants->get_mat_ptr(matrix), x, beta, y);
      break;
    case DGConstants::DSW:
      op2_gemv_np_np(mesh, transpose, alpha, constants->get_mat_ptr(matrix), x, beta, y);
      break;
    case DGConstants::CUB_V:
      op2_gemv_cub_np_np(mesh, transpose, alpha, constants->get_mat_ptr(matrix), x, beta, y);
      break;
    case DGConstants::INV_MASS:
      op2_gemv_np_np(mesh, transpose, alpha, constants->get_mat_ptr(matrix), x, beta, y);
      break;
    case DGConstants::LIFT:
      op2_gemv_lift(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::MASS:
      op2_gemv_np_np(mesh, transpose, alpha, constants->get_mat_ptr(matrix), x, beta, y);
      break;
    case DGConstants::V:
      op2_gemv_np_np(mesh, transpose, alpha, constants->get_mat_ptr(matrix), x, beta, y);
      break;
    case DGConstants::INV_V:
      op2_gemv_np_np(mesh, transpose, alpha, constants->get_mat_ptr(matrix), x, beta, y);
      break;
    case DGConstants::CUB_VDR:
      op2_gemv_cub_np_np(mesh, transpose, alpha, constants->get_mat_ptr(matrix), x, beta, y);
      break;
    case DGConstants::CUB_VDS:
      op2_gemv_cub_np_np(mesh, transpose, alpha, constants->get_mat_ptr(matrix), x, beta, y);
      break;
    default:
      std::cerr << "op2_gemv call not implemented for this matrix ... exiting" << std::endl;
      exit(2);
  }
}
