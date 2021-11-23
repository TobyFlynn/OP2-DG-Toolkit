#include "dg_op2_blas.h"

#include "op_seq.h"

#include <iostream>

void op2_gemv_gauss_interp(DGMesh *mesh, bool transpose, const double alpha,
                           op_dat x, const double beta, op_dat y) {
  if(transpose) {
    op_par_loop(gemv_gauss_interp, "gemv_gauss_interp", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&transpose, 1, "bool", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(gInterp_g, DG_ORDER * DG_G_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_G_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
  } else {
    op_par_loop(gemv_gauss_interp, "gemv_gauss_interp", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&transpose, 1, "bool", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(gInterp_g, DG_ORDER * DG_G_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_G_NP, "double", OP_RW));
  }
}

void op2_gemv_cub_dr(DGMesh *mesh, bool transpose, const double alpha, op_dat x,
                     const double beta, op_dat y) {
  if(transpose) {
    op_par_loop(gemv_cub_np_np, "gemv_cub_np_np", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&transpose, 1, "bool", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(cubDr_g, DG_ORDER * DG_CUB_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
  } else {
    op_par_loop(gemv_cub_np_np, "gemv_cub_np_np", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&transpose, 1, "bool", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(cubDr_g, DG_ORDER * DG_CUB_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_CUB_NP, "double", OP_RW));
  }
}

void op2_gemv_cub_ds(DGMesh *mesh, bool transpose, const double alpha, op_dat x,
                     const double beta, op_dat y) {
  if(transpose) {
    op_par_loop(gemv_cub_np_np, "gemv_cub_np_np", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&transpose, 1, "bool", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(cubDs_g, DG_ORDER * DG_CUB_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
  } else {
    op_par_loop(gemv_cub_np_np, "gemv_cub_np_np", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&transpose, 1, "bool", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(cubDs_g, DG_ORDER * DG_CUB_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_CUB_NP, "double", OP_RW));
  }
}

void op2_gemv_dr(DGMesh *mesh, bool transpose, const double alpha, op_dat x,
                 const double beta, op_dat y) {
  op_par_loop(gemv_np_np, "gemv_np_np", mesh->cells,
              op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(&transpose, 1, "bool", OP_READ),
              op_arg_gbl(&alpha, 1, "double", OP_READ),
              op_arg_gbl(&beta,  1, "double", OP_READ),
              op_arg_gbl(Dr_g, DG_ORDER * DG_NP * DG_NP, "double", OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
}

void op2_gemv_ds(DGMesh *mesh, bool transpose, const double alpha, op_dat x,
                 const double beta, op_dat y) {
  op_par_loop(gemv_np_np, "gemv_np_np", mesh->cells,
              op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(&transpose, 1, "bool", OP_READ),
              op_arg_gbl(&alpha, 1, "double", OP_READ),
              op_arg_gbl(&beta,  1, "double", OP_READ),
              op_arg_gbl(Ds_g, DG_ORDER * DG_NP * DG_NP, "double", OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
}

void op2_gemv_drw(DGMesh *mesh, bool transpose, const double alpha, op_dat x,
                 const double beta, op_dat y) {
  op_par_loop(gemv_np_np, "gemv_np_np", mesh->cells,
              op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(&transpose, 1, "bool", OP_READ),
              op_arg_gbl(&alpha, 1, "double", OP_READ),
              op_arg_gbl(&beta,  1, "double", OP_READ),
              op_arg_gbl(Drw_g, DG_ORDER * DG_NP * DG_NP, "double", OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
}

void op2_gemv_dsw(DGMesh *mesh, bool transpose, const double alpha, op_dat x,
                 const double beta, op_dat y) {
  op_par_loop(gemv_np_np, "gemv_np_np", mesh->cells,
              op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(&transpose, 1, "bool", OP_READ),
              op_arg_gbl(&alpha, 1, "double", OP_READ),
              op_arg_gbl(&beta,  1, "double", OP_READ),
              op_arg_gbl(Dsw_g, DG_ORDER * DG_NP * DG_NP, "double", OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
}

void op2_gemv_cub_v(DGMesh *mesh, bool transpose, const double alpha, op_dat x,
                    const double beta, op_dat y) {
  if(transpose) {
    op_par_loop(gemv_cub_np_np, "gemv_cub_np_np", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&transpose, 1, "bool", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(cubV_g, DG_ORDER * DG_CUB_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
  } else {
    op_par_loop(gemv_cub_np_np, "gemv_cub_np_np", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&transpose, 1, "bool", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(cubV_g, DG_ORDER * DG_CUB_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_CUB_NP, "double", OP_RW));
  }
}

void op2_gemv_inv_mass(DGMesh *mesh, bool transpose, const double alpha,
                       op_dat x, const double beta, op_dat y) {
  op_par_loop(gemv_np_np, "gemv_np_np", mesh->cells,
              op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(&transpose, 1, "bool", OP_READ),
              op_arg_gbl(&alpha, 1, "double", OP_READ),
              op_arg_gbl(&beta,  1, "double", OP_READ),
              op_arg_gbl(invMass_g, DG_ORDER * DG_NP * DG_NP, "double", OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
}

void op2_gemv_lift(DGMesh *mesh, bool transpose, const double alpha, op_dat x,
                   const double beta, op_dat y) {
  if(transpose) {
    op_par_loop(gemv_lift, "gemv_lift", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&transpose, 1, "bool", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(lift_g, DG_ORDER * 3 * DG_NPF * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, 3 * DG_NPF, "double", OP_RW));
  } else {
    op_par_loop(gemv_lift, "gemv_lift", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&transpose, 1, "bool", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(lift_g, DG_ORDER * 3 * DG_NPF * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, 3 * DG_NPF, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
  }
}

void op2_gemv_mass(DGMesh *mesh, bool transpose, const double alpha, op_dat x,
                   const double beta, op_dat y) {
  op_par_loop(gemv_np_np, "gemv_np_np", mesh->cells,
              op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(&transpose, 1, "bool", OP_READ),
              op_arg_gbl(&alpha, 1, "double", OP_READ),
              op_arg_gbl(&beta,  1, "double", OP_READ),
              op_arg_gbl(mass_g, DG_ORDER * DG_NP * DG_NP, "double", OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
}

void op2_gemv_inv_v(DGMesh *mesh, bool transpose, const double alpha, op_dat x,
                    const double beta, op_dat y) {
  op_par_loop(gemv_np_np, "gemv_np_np", mesh->cells,
              op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(&transpose, 1, "bool", OP_READ),
              op_arg_gbl(&alpha, 1, "double", OP_READ),
              op_arg_gbl(&beta,  1, "double", OP_READ),
              op_arg_gbl(invV_g, DG_ORDER * DG_NP * DG_NP, "double", OP_READ),
              op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
              op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
}

void op2_gemv_cub_vdr(DGMesh *mesh, bool transpose, const double alpha, op_dat x,
                      const double beta, op_dat y) {
  if(transpose) {
    op_par_loop(gemv_cub_np_np, "gemv_cub_np_np", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&transpose, 1, "bool", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(cubVDr_g, DG_ORDER * DG_CUB_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
  } else {
    op_par_loop(gemv_cub_np_np, "gemv_cub_np_np", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&transpose, 1, "bool", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(cubVDr_g, DG_ORDER * DG_CUB_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_CUB_NP, "double", OP_RW));
  }
}

void op2_gemv_cub_vds(DGMesh *mesh, bool transpose, const double alpha, op_dat x,
                      const double beta, op_dat y) {
  if(transpose) {
    op_par_loop(gemv_cub_np_np, "gemv_cub_np_np", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&transpose, 1, "bool", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(cubVDs_g, DG_ORDER * DG_CUB_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_CUB_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_NP, "double", OP_RW));
  } else {
    op_par_loop(gemv_cub_np_np, "gemv_cub_np_np", mesh->cells,
                op_arg_dat(mesh->order, -1, OP_ID, 1, "int", OP_READ),
                op_arg_gbl(&transpose, 1, "bool", OP_READ),
                op_arg_gbl(&alpha, 1, "double", OP_READ),
                op_arg_gbl(&beta,  1, "double", OP_READ),
                op_arg_gbl(cubVDs_g, DG_ORDER * DG_CUB_NP * DG_NP, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, DG_NP, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, DG_CUB_NP, "double", OP_RW));
  }
}

void op2_gemv(DGMesh *mesh, bool transpose, const double alpha,
              DGConstants::Constant_Matrix matrix, op_dat x, const double beta,
              op_dat y) {
  switch(matrix) {
    case DGConstants::GAUSS_INTERP:
      op2_gemv_gauss_interp(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::CUB_DR:
      op2_gemv_cub_dr(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::CUB_DS:
      op2_gemv_cub_ds(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::DR:
      op2_gemv_dr(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::DS:
      op2_gemv_ds(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::DRW:
      op2_gemv_drw(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::DSW:
      op2_gemv_dsw(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::CUB_V:
      op2_gemv_cub_v(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::INV_MASS:
      op2_gemv_inv_mass(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::LIFT:
      op2_gemv_lift(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::MASS:
      op2_gemv_mass(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::INV_V:
      op2_gemv_inv_v(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::CUB_VDR:
      op2_gemv_cub_vdr(mesh, transpose, alpha, x, beta, y);
      break;
    case DGConstants::CUB_VDS:
      op2_gemv_cub_vds(mesh, transpose, alpha, x, beta, y);
      break;
    default:
      std::cerr << "op2_gemv call not implemented for this matrix ... exiting" << std::endl;
      exit(2);
  }
}
