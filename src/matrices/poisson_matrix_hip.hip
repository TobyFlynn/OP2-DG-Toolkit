#include "dg_matrices/poisson_matrix.h"

#ifdef DG_MPI
#include "mpi_helper_func.h"
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "dg_utils.h"
#include "dg_global_constants/dg_global_constants_2d.h"
#include "dg_abort.h"

void get_num_nodes(const int N, int *Np, int *Nfp) {
  #if DG_DIM == 2
  DGUtils::numNodes2D(N, Np, Nfp);
  #elif DG_DIM == 3
  DGUtils::numNodes3D(N, Np, Nfp);
  #endif
}

void PoissonMatrix::set_glb_ind() {
  DG_MAT_IND_TYPE unknowns = getUnknowns();
  DG_MAT_IND_TYPE global_ind = 0;
  #ifdef DG_MPI
  global_ind = get_global_mat_start_ind(unknowns);
  #endif
  op_arg args[] = {
    op_arg_dat(glb_ind, -1, OP_ID, 1, DG_MAT_IND_TYPE_STR, OP_WRITE)
  };
  op_mpi_halo_exchanges_cuda(_mesh->cells, 1, args);

  const int setSize = _mesh->cells->size;
  DG_MAT_IND_TYPE *data_ptr = (DG_MAT_IND_TYPE *)malloc(setSize * sizeof(DG_MAT_IND_TYPE));
  hipMemcpy(data_ptr, glb_ind->data_d, setSize * sizeof(DG_MAT_IND_TYPE), hipMemcpyDeviceToHost);

  DG_MAT_IND_TYPE ind = global_ind;
  for(int i = 0; i < _mesh->cells->size; i++) {
    data_ptr[i] = ind;
    ind += DG_NP;
  }

  hipMemcpy(glb_ind->data_d, data_ptr, setSize * sizeof(DG_MAT_IND_TYPE), hipMemcpyHostToDevice);

  op_mpi_set_dirtybit_cuda(1, args);
  free(data_ptr);
}

void PoissonMatrix::setPETScMatrix() {
  #ifdef DG_OP2_SOA
  dg_abort("setPETScMatrix not implemented for SoA");
  #endif
  if(!petscMatInit) {
    MatCreate(PETSC_COMM_WORLD, &pMat);
    petscMatInit = true;
    DG_MAT_IND_TYPE unknowns = getUnknowns();
    MatSetSizes(pMat, unknowns, unknowns, PETSC_DECIDE, PETSC_DECIDE);

    #ifdef DG_MPI
    #ifdef PETSC_COMPILED_WITH_HIP
    MatSetType(pMat, MATMPIAIJHIPSPARSE);
    #else
    MatSetType(pMat, MATMPIAIJ);
    #endif
    MatMPIAIJSetPreallocation(pMat, DG_NP * (DG_NUM_FACES + 1), NULL, 0, NULL);
    #else
    #ifdef PETSC_COMPILED_WITH_HIP
    MatSetType(pMat, MATSEQAIJHIPSPARSE);
    #else
    MatSetType(pMat, MATSEQAIJ);
    #endif
    MatSeqAIJSetPreallocation(pMat, DG_NP * (DG_NUM_FACES + 1), NULL);
    #endif
    MatSetOption(pMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  }
  // Add cubature OP to Poisson matrix
  op_arg args[] = {
    op_arg_dat(op1, -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
    op_arg_dat(glb_ind, -1, OP_ID, 1, DG_MAT_IND_TYPE_STR, OP_READ)
  };
  op_mpi_halo_exchanges_cuda(_mesh->cells, 2, args);

  const int setSize = _mesh->cells->size;
  DG_FP *op1_data = (DG_FP *)malloc(DG_NP * DG_NP * setSize * sizeof(DG_FP));
  DG_MAT_IND_TYPE *glb = (DG_MAT_IND_TYPE *)malloc(setSize * sizeof(DG_MAT_IND_TYPE));
  hipMemcpy(op1_data, op1->data_d, setSize * DG_NP * DG_NP * sizeof(DG_FP), hipMemcpyDeviceToHost);
  hipMemcpy(glb, glb_ind->data_d, setSize * sizeof(DG_MAT_IND_TYPE), hipMemcpyDeviceToHost);
  op_mpi_set_dirtybit_cuda(2, args);

  #ifdef DG_COL_MAJ
  MatSetOption(pMat, MAT_ROW_ORIENTED, PETSC_FALSE);
  #else
  MatSetOption(pMat, MAT_ROW_ORIENTED, PETSC_TRUE);
  #endif

  for(int i = 0; i < setSize; i++) {
    DG_MAT_IND_TYPE currentRow = glb[i];
    DG_MAT_IND_TYPE currentCol = glb[i];
    PetscInt idxm[DG_NP], idxn[DG_NP];
    for(DG_MAT_IND_TYPE n = 0; n < DG_NP; n++) {
      idxm[n] = static_cast<PetscInt>(currentRow + n);
      idxn[n] = static_cast<PetscInt>(currentCol + n);
    }

    MatSetValues(pMat, DG_NP, idxm, DG_NP, idxn, &op1_data[i * DG_NP * DG_NP], INSERT_VALUES);
  }

  free(op1_data);
  free(glb);

  op_arg edge_args[] = {
    op_arg_dat(op2[0], -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
    op_arg_dat(op2[1], -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
    op_arg_dat(glb_indL, -1, OP_ID, 1, DG_MAT_IND_TYPE_STR, OP_READ),
    op_arg_dat(glb_indR, -1, OP_ID, 1, DG_MAT_IND_TYPE_STR, OP_READ)
  };
  op_mpi_halo_exchanges_cuda(_mesh->faces, 4, edge_args);
  DG_FP *op2L_data = (DG_FP *)malloc(DG_NP * DG_NP * _mesh->faces->size * sizeof(DG_FP));
  DG_FP *op2R_data = (DG_FP *)malloc(DG_NP * DG_NP * _mesh->faces->size * sizeof(DG_FP));
  DG_MAT_IND_TYPE *glb_l = (DG_MAT_IND_TYPE *)malloc(_mesh->faces->size * sizeof(DG_MAT_IND_TYPE));
  DG_MAT_IND_TYPE *glb_r = (DG_MAT_IND_TYPE *)malloc(_mesh->faces->size * sizeof(DG_MAT_IND_TYPE));

  hipMemcpy(op2L_data, op2[0]->data_d, DG_NP * DG_NP * _mesh->faces->size * sizeof(DG_FP), hipMemcpyDeviceToHost);
  hipMemcpy(op2R_data, op2[1]->data_d, DG_NP * DG_NP * _mesh->faces->size * sizeof(DG_FP), hipMemcpyDeviceToHost);
  hipMemcpy(glb_l, glb_indL->data_d, _mesh->faces->size * sizeof(DG_MAT_IND_TYPE), hipMemcpyDeviceToHost);
  hipMemcpy(glb_r, glb_indR->data_d, _mesh->faces->size * sizeof(DG_MAT_IND_TYPE), hipMemcpyDeviceToHost);

  // Add Gauss OP and OPf to Poisson matrix
  for(int i = 0; i < _mesh->faces->size; i++) {
    DG_MAT_IND_TYPE leftRow = glb_l[i];
    DG_MAT_IND_TYPE rightRow = glb_r[i];

    PetscInt idxl[DG_NP], idxr[DG_NP];
    for(DG_MAT_IND_TYPE n = 0; n < DG_NP; n++) {
      idxl[n] = static_cast<PetscInt>(leftRow + n);
      idxr[n] = static_cast<PetscInt>(rightRow + n);
    }

    MatSetValues(pMat, DG_NP, idxl, DG_NP, idxr, &op2L_data[i * DG_NP * DG_NP], INSERT_VALUES);
    MatSetValues(pMat, DG_NP, idxr, DG_NP, idxl, &op2R_data[i * DG_NP * DG_NP], INSERT_VALUES);
  }

  free(op2L_data);
  free(op2R_data);
  free(glb_l);
  free(glb_r);

  op_mpi_set_dirtybit_cuda(4, edge_args);

  MatAssemblyBegin(pMat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(pMat, MAT_FINAL_ASSEMBLY);
}
