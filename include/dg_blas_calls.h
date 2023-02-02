#ifndef __DG_BLAS_CALLS_H
#define __DG_BLAS_CALLS_H

#include "op_seq.h"
#include "dg_mesh/dg_mesh_2d.h"
#include "dg_constants/dg_constants.h"

#ifdef OP2_DG_CUDA
#include "cublas_v2.h"
extern cublasHandle_t handle;
#endif

void inv_blas(DGMesh2D *mesh, op_dat in, op_dat out);

void init_blas();

void destroy_blas();

#endif
