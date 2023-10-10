#include "dg_matrices/poisson_matrix.h"

#include "op_seq.h"

#ifdef DG_MPI
#include "mpi_helper_func.h"
#include <iostream>
#include "op_mpi_core.h"
#endif

#include "dg_utils.h"
#include "dg_global_constants/dg_global_constants_2d.h"

void get_num_nodes(const int N, int *Np, int *Nfp) {
  #if DG_DIM == 2
  DGUtils::numNodes2D(N, Np, Nfp);
  #elif DG_DIM == 3
  DGUtils::numNodes3D(N, Np, Nfp);
  #endif
}

DG_MAT_IND_TYPE PoissonMatrix::getUnknowns() {
  op_arg op2_args[] = {
    op_arg_dat(_mesh->order, -1, OP_ID, 1, "int", OP_READ)
  };
  op_mpi_halo_exchanges(_mesh->order->set, 1, op2_args);
  const int setSize = _mesh->order->set->size;
  const int *tempOrder = (int *)_mesh->order->data;
  DG_MAT_IND_TYPE unknowns = 0;
  #pragma omp parallel for reduction(+:unknowns)
  for(int i = 0; i < setSize; i++) {
    int Np, Nfp;
    get_num_nodes(tempOrder[i], &Np, &Nfp);
    unknowns += Np;
  }
  op_mpi_set_dirtybit(1, op2_args);
  return unknowns;
}

void PoissonMatrix::set_glb_ind() {
  DG_MAT_IND_TYPE unknowns = getUnknowns();
  DG_MAT_IND_TYPE global_ind = 0;
  #ifdef DG_MPI
  global_ind = get_global_mat_start_ind(unknowns);
  #endif
  op_arg args[] = {
    op_arg_dat(_mesh->order, -1, OP_ID, 1, "int", OP_READ),
    op_arg_dat(glb_ind, -1, OP_ID, 1, DG_MAT_IND_TYPE_STR, OP_WRITE)
  };
  op_mpi_halo_exchanges(_mesh->cells, 2, args);

  const int *p = (int *)_mesh->order->data;
  DG_MAT_IND_TYPE *data_ptr = (DG_MAT_IND_TYPE *)glb_ind->data;
  DG_MAT_IND_TYPE ind = global_ind;
  for(int i = 0; i < _mesh->cells->size; i++) {
    int Np, Nfp;
    get_num_nodes(p[i], &Np, &Nfp);
    data_ptr[i] = ind;
    ind += Np;
  }

  op_mpi_set_dirtybit(2, args);
}

void PoissonMatrix::setPETScMatrix() {
  if(!petscMatInit) {
    MatCreate(PETSC_COMM_WORLD, &pMat);
    petscMatInit = true;
    DG_MAT_IND_TYPE unknowns = getUnknowns();
    MatSetSizes(pMat, unknowns, unknowns, PETSC_DECIDE, PETSC_DECIDE);

    #ifdef DG_MPI
    MatSetType(pMat, MATMPIAIJ);
    MatMPIAIJSetPreallocation(pMat, DG_NP * (DG_NUM_FACES + 1), NULL, 0, NULL);
    #else
    MatSetType(pMat, MATSEQAIJ);
    MatSeqAIJSetPreallocation(pMat, DG_NP * (DG_NUM_FACES + 1), NULL);
    #endif
    MatSetOption(pMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  }

  // Add cubature OP to Poisson matrix
  op_arg args[] = {
    op_arg_dat(op1, -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
    op_arg_dat(glb_ind, -1, OP_ID, 1, DG_MAT_IND_TYPE_STR, OP_READ),
    op_arg_dat(_mesh->order, -1, OP_ID, 1, "int", OP_READ)
  };
  op_mpi_halo_exchanges(_mesh->cells, 3, args);
  const DG_FP *op1_data = (DG_FP *)op1->data;
  const DG_MAT_IND_TYPE *glb = (DG_MAT_IND_TYPE *)glb_ind->data;
  const int *p = (int *)_mesh->order->data;

  #ifdef DG_COL_MAJ
  MatSetOption(pMat, MAT_ROW_ORIENTED, PETSC_FALSE);
  #else
  MatSetOption(pMat, MAT_ROW_ORIENTED, PETSC_TRUE);
  #endif

  for(int i = 0; i < _mesh->cells->size; i++) {
    int Np, Nfp;
    get_num_nodes(p[i], &Np, &Nfp);
    DG_MAT_IND_TYPE currentRow = glb[i];
    DG_MAT_IND_TYPE currentCol = glb[i];

    PetscInt idxm[DG_NP], idxn[DG_NP];
    for(DG_MAT_IND_TYPE n = 0; n < DG_NP; n++) {
      idxm[n] = static_cast<PetscInt>(currentRow + n);
      idxn[n] = static_cast<PetscInt>(currentCol + n);
    }

    MatSetValues(pMat, Np, idxm, Np, idxn, &op1_data[i * DG_NP * DG_NP], INSERT_VALUES);
  }

  op_mpi_set_dirtybit(3, args);

  op_arg edge_args[] = {
    op_arg_dat(op2[0], -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
    op_arg_dat(op2[1], -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
    op_arg_dat(glb_indL, -1, OP_ID, 1, DG_MAT_IND_TYPE_STR, OP_READ),
    op_arg_dat(glb_indR, -1, OP_ID, 1, DG_MAT_IND_TYPE_STR, OP_READ),
    op_arg_dat(orderL, -1, OP_ID, 1, "int", OP_READ),
    op_arg_dat(orderR, -1, OP_ID, 1, "int", OP_READ)
  };
  op_mpi_halo_exchanges(_mesh->faces, 6, edge_args);

  const DG_FP *op2L_data = (DG_FP *)op2[0]->data;
  const DG_FP *op2R_data = (DG_FP *)op2[1]->data;
  const DG_MAT_IND_TYPE *glb_l = (DG_MAT_IND_TYPE *)glb_indL->data;
  const DG_MAT_IND_TYPE *glb_r = (DG_MAT_IND_TYPE *)glb_indR->data;
  const int *p_l = (int *)orderL->data;
  const int *p_r = (int *)orderR->data;

  // Add Gauss OP and OPf to Poisson matrix
  for(int i = 0; i < _mesh->faces->size; i++) {
    DG_MAT_IND_TYPE leftRow = glb_l[i];
    DG_MAT_IND_TYPE rightRow = glb_r[i];
    int NpL, NpR, Nfp;
    get_num_nodes(p_l[i], &NpL, &Nfp);
    get_num_nodes(p_r[i], &NpR, &Nfp);

    PetscInt idxl[DG_NP], idxr[DG_NP];
    for(DG_MAT_IND_TYPE n = 0; n < DG_NP; n++) {
      idxl[n] = static_cast<PetscInt>(leftRow + n);
      idxr[n] = static_cast<PetscInt>(rightRow + n);
    }

    MatSetValues(pMat, NpL, idxl, NpR, idxr, &op2L_data[i * DG_NP * DG_NP], INSERT_VALUES);
    MatSetValues(pMat, NpR, idxr, NpL, idxl, &op2R_data[i * DG_NP * DG_NP], INSERT_VALUES);
  }

  op_mpi_set_dirtybit(6, edge_args);

  MatAssemblyBegin(pMat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(pMat, MAT_FINAL_ASSEMBLY);
}
