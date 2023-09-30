#include "dg_matrices/poisson_matrix.h"

#ifdef DG_MPI
#include "mpi_helper_func.h"
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "dg_utils.h"
#include "dg_global_constants/dg_global_constants_2d.h"

void get_num_nodes(const int N, int *Np, int *Nfp) {
  #if DG_DIM == 2
  DGUtils::numNodes2D(N, Np, Nfp);
  #elif DG_DIM == 3
  DGUtils::numNodes3D(N, Np, Nfp);
  #endif
}

int PoissonMatrix::getUnknowns() {
  return DG_NP * _mesh->cells->size;
}

void PoissonMatrix::set_glb_ind() {
  int unknowns = getUnknowns();
  int global_ind = 0;
  #ifdef DG_MPI
  global_ind = get_global_mat_start_ind(unknowns);
  #endif
  op_arg args[] = {
    op_arg_dat(glb_ind, -1, OP_ID, 1, "int", OP_WRITE)
  };
  op_mpi_halo_exchanges_cuda(_mesh->cells, 1, args);

  const int setSize = _mesh->cells->size;
  int *data_ptr = (int *)malloc(setSize * sizeof(int));
  hipMemcpy(data_ptr, glb_ind->data_d, setSize * sizeof(int), hipMemcpyDeviceToHost);

  int ind = global_ind;
  for(int i = 0; i < _mesh->cells->size; i++) {
    data_ptr[i] = ind;
    ind += DG_NP;
  }

  hipMemcpy(glb_ind->data_d, data_ptr, setSize * sizeof(int), hipMemcpyHostToDevice);

  op_mpi_set_dirtybit_cuda(1, args);
  free(data_ptr);
}

void PoissonMatrix::setPETScMatrix() {
  #ifdef DG_OP2_SOA
  throw std::runtime_error("setPETScMatrix not implemented for SoA");
  #endif
  if(!petscMatInit) {
    MatCreate(PETSC_COMM_WORLD, &pMat);
    petscMatInit = true;
    int unknowns = getUnknowns();
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
    op_arg_dat(glb_ind, -1, OP_ID, 1, "int", OP_READ)
  };
  op_mpi_halo_exchanges_cuda(_mesh->cells, 2, args);

  const int setSize = _mesh->cells->size;
  DG_FP *op1_data = (DG_FP *)malloc(DG_NP * DG_NP * setSize * sizeof(DG_FP));
  int *glb   = (int *)malloc(setSize * sizeof(int));
  hipMemcpy(op1_data, op1->data_d, setSize * DG_NP * DG_NP * sizeof(DG_FP), hipMemcpyDeviceToHost);
  hipMemcpy(glb, glb_ind->data_d, setSize * sizeof(int), hipMemcpyDeviceToHost);
  op_mpi_set_dirtybit_cuda(2, args);

  #ifdef DG_COL_MAJ
  MatSetOption(pMat, MAT_ROW_ORIENTED, PETSC_FALSE);
  #else
  MatSetOption(pMat, MAT_ROW_ORIENTED, PETSC_TRUE);
  #endif

  for(int i = 0; i < setSize; i++) {
    int currentRow = glb[i];
    int currentCol = glb[i];
    int idxm[DG_NP], idxn[DG_NP];
    for(int n = 0; n < DG_NP; n++) {
      idxm[n] = currentRow + n;
      idxn[n] = currentCol + n;
    }

    MatSetValues(pMat, DG_NP, idxm, DG_NP, idxn, &op1_data[i * DG_NP * DG_NP], INSERT_VALUES);
  }

  free(op1_data);
  free(glb);

  op_arg edge_args[] = {
    op_arg_dat(op2[0], -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
    op_arg_dat(op2[1], -1, OP_ID, DG_NP * DG_NP, DG_FP_STR, OP_READ),
    op_arg_dat(glb_indL, -1, OP_ID, 1, "int", OP_READ),
    op_arg_dat(glb_indR, -1, OP_ID, 1, "int", OP_READ)
  };
  op_mpi_halo_exchanges_cuda(_mesh->faces, 4, edge_args);
  DG_FP *op2L_data = (DG_FP *)malloc(DG_NP * DG_NP * _mesh->faces->size * sizeof(DG_FP));
  DG_FP *op2R_data = (DG_FP *)malloc(DG_NP * DG_NP * _mesh->faces->size * sizeof(DG_FP));
  int *glb_l = (int *)malloc(_mesh->faces->size * sizeof(int));
  int *glb_r = (int *)malloc(_mesh->faces->size * sizeof(int));

  hipMemcpy(op2L_data, op2[0]->data_d, DG_NP * DG_NP * _mesh->faces->size * sizeof(DG_FP), hipMemcpyDeviceToHost);
  hipMemcpy(op2R_data, op2[1]->data_d, DG_NP * DG_NP * _mesh->faces->size * sizeof(DG_FP), hipMemcpyDeviceToHost);
  hipMemcpy(glb_l, glb_indL->data_d, _mesh->faces->size * sizeof(int), hipMemcpyDeviceToHost);
  hipMemcpy(glb_r, glb_indR->data_d, _mesh->faces->size * sizeof(int), hipMemcpyDeviceToHost);

  // Add Gauss OP and OPf to Poisson matrix
  for(int i = 0; i < _mesh->faces->size; i++) {
    int leftRow = glb_l[i];
    int rightRow = glb_r[i];

    int idxl[DG_NP], idxr[DG_NP];
    for(int n = 0; n < DG_NP; n++) {
      idxl[n] = leftRow + n;
      idxr[n] = rightRow + n;
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
