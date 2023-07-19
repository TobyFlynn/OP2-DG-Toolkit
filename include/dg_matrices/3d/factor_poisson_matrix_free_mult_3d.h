#ifndef __DG_FACTOR_POISSON_MATRIX_FREE_MULT_3D_H
#define __DG_FACTOR_POISSON_MATRIX_FREE_MULT_3D_H

#include "op_seq.h"
#include "dg_mesh/dg_mesh_3d.h"
#include "poisson_matrix_free_mult_3d.h"

class FactorPoissonMatrixFreeMult3D : public PoissonMatrixFreeMult3D {
public:
  FactorPoissonMatrixFreeMult3D(DGMesh3D *m);

  // op_dat bc_types - 0 for Dirichlet, 1 for Neumann
  virtual void mat_free_apply_bc(op_dat rhs, op_dat bc) override;
  virtual void mat_free_mult(op_dat in, op_dat out) override;
  virtual void mat_free_set_factor(op_dat f);
  virtual void mat_free_mult_sp(op_dat in, op_dat out) override;

protected:
  op_dat mat_free_factor, mat_free_factor_copy, mat_free_gtau, mat_free_factor_copy_sp;

  void check_current_order();
  void calc_tau();

private:
  int factor_order, current_order;
};

#endif
