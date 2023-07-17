#ifndef __DG_FACTOR_POISSON_MATRIX_3D_H
#define __DG_FACTOR_POISSON_MATRIX_3D_H

#include "poisson_matrix_3d.h"

class FactorPoissonMatrix3D : public PoissonMatrix3D {
public:
  FactorPoissonMatrix3D(DGMesh3D *m);

  void set_factor(op_dat f);

protected:
  virtual void calc_op1() override;
  virtual void calc_op2() override;
  virtual void calc_opbc() override;

  op_dat factor;
};

#endif
