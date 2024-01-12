#ifndef __DG_POISSON_MATRIX_FREE_BLOCK_DIAG_H
#define __DG_POISSON_MATRIX_FREE_BLOCK_DIAG_H

#include "dg_compiler_defs.h"

#include "op_seq.h"

#include "petscvec.h"
#include "petscksp.h"

#include "dg_mesh/dg_mesh.h"
#include "poisson_matrix.h"

class PoissonMatrixFreeBlockDiag : public PoissonMatrix {
public:
  virtual void calc_mat_partial() = 0;
  virtual void calc_mat() override;
  virtual void mult(op_dat in, op_dat out) override;
  virtual void multJacobi(op_dat in, op_dat out) override;
  virtual bool getPETScMat(Mat** mat) override;

  op_dat block_diag;

protected:
  virtual void setPETScMatrix() override;
  virtual void calc_glb_ind() override;
};

#endif
