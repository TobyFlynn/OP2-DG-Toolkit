#ifndef __DG_MM_POISSON_MATRIX_FREE_OVER_INT_2D_H
#define __DG_MM_POISSON_MATRIX_FREE_OVER_INT_2D_H

#include "factor_poisson_matrix_free_over_int_2d.h"

class FactorMMPoissonMatrixFreeOverInt2D : public FactorPoissonMatrixFreeOverInt2D {
public:
  FactorMMPoissonMatrixFreeOverInt2D(DGMesh2D *m);

  void set_mm_factor(op_dat f);
  virtual void mult(op_dat in, op_dat out) override;

private:
  op_dat mm_factor;
};

#endif
