#include "dg_matrices/poisson_matrix.h"

#include "op_seq.h"

#include "timing.h"
#include "dg_abort.h"

extern Timing *timer;

void PoissonMatrix::set_bc_types(op_dat bc_ty) {
  bc_types = bc_ty;
}

bool PoissonMatrix::getPETScMat(Mat** mat) {
  timer->startTimer("PoissonMatrix - getPETScMat");
  bool reset = false;
  if(petscMatResetRequired) {
    setPETScMatrix();
    petscMatResetRequired = false;
    *mat = &pMat;
    reset = true;
  }
  timer->endTimer("PoissonMatrix - getPETScMat");

  return reset;
}

void PoissonMatrix::mult(op_dat in, op_dat out) {
  timer->startTimer("PoissonMatrix - mult");
  timer->startTimer("PoissonMatrix - mult Cells");
  op_par_loop(poisson_mult_cells, "poisson_mult_cells", _mesh->cells,
              op_arg_gbl(&_mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(in,  -1, OP_ID, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op1, -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(out, -1, OP_ID, DG_NP, DG_FP_STR, OP_WRITE));
  timer->endTimer("PoissonMatrix - mult Cells");
  timer->startTimer("PoissonMatrix - mult Faces");
  op_par_loop(poisson_mult_faces, "poisson_mult_faces", _mesh->faces,
              op_arg_gbl(&_mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(in,      0, _mesh->face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op2[0], -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(out,     0, _mesh->face2cells, DG_NP, DG_FP_STR, OP_INC),
              op_arg_dat(in,      1, _mesh->face2cells, DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(op2[1], -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(out,     1, _mesh->face2cells, DG_NP, DG_FP_STR, OP_INC));
  timer->endTimer("PoissonMatrix - mult Faces");
  timer->endTimer("PoissonMatrix - mult");
}

void PoissonMatrix::mult_sp(op_dat in, op_dat out) {
  dg_abort("mult_sp not implemented");
}

void PoissonMatrix::multJacobi(op_dat in, op_dat out) {
  timer->startTimer("PoissonMatrix - multJacobi");
  mult(in, out);

  op_par_loop(poisson_mult_jacobi, "poisson_mult_jacobi", _mesh->cells,
              op_arg_gbl(&_mesh->order_int, 1, "int", OP_READ),
              op_arg_dat(op1, -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
              op_arg_dat(out, -1, OP_ID, DG_NP, DG_FP_STR, OP_RW));
  timer->endTimer("PoissonMatrix - multJacobi");
}

void PoissonMatrix::multJacobi_sp(op_dat in, op_dat out) {
  dg_abort("multJacobi_sp not implemented");
}

DG_MAT_IND_TYPE PoissonMatrix::getUnknowns() {
  return DG_NP * static_cast<DG_MAT_IND_TYPE>(_mesh->cells->size);
}
