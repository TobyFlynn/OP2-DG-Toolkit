#ifndef __DG_POISSON_MATRIX_FREE_BLOCK_DIAG_2D_H
#define __DG_POISSON_MATRIX_FREE_BLOCK_DIAG_2D_H

#include "op_seq.h"
#include "dg_mesh/dg_mesh_2d.h"
#include "../poisson_matrix_free_block_diag.h"
#include "poisson_matrix_free_mult_2d.h"

class PoissonMatrixFreeBlockDiag2D : public PoissonMatrixFreeBlockDiag, public PoissonMatrixFreeMult2D {
public:
  PoissonMatrixFreeBlockDiag2D(DGMesh2D *m);

  // op_dat bc_types - 0 for Dirichlet, 1 for Neumann
  virtual void set_bc_types(op_dat bc_ty) override;
  virtual void apply_bc(op_dat rhs, op_dat bc) override;
  virtual void mult(op_dat in, op_dat out) override;
  virtual void mult_sp(op_dat in, op_dat out) override;
  virtual void multJacobi_sp(op_dat in, op_dat out) override;
  virtual void calc_mat_partial() override;

protected:
  virtual void calc_op1() override;
  virtual void calc_op2() override;
  virtual void calc_opbc() override;
};

#endif
