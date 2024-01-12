#include "dg_matrices/poisson_coarse_matrix.h"

#ifdef DG_MPI
#include "mpi_helper_func.h"
#endif

#include <vector>
#include <algorithm>
#include <map>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "op2_utils.h"
#include "dg_utils.h"
#include "dg_global_constants/dg_global_constants_2d.h"
#include "dg_abort.h"

#include "timing.h"
extern Timing *timer;

void PoissonCoarseMatrix::set_glb_ind() {
  DG_MAT_IND_TYPE unknowns = getUnknowns();
  DG_MAT_IND_TYPE global_ind = 0;
  #ifdef DG_MPI
  global_ind = get_global_mat_start_ind(unknowns);
  #endif
  op_arg args[] = {
    op_arg_dat(glb_ind, -1, OP_ID, 1, DG_MAT_IND_TYPE_STR, OP_WRITE)
  };
  op_mpi_halo_exchanges_grouped(_mesh->cells, 1, args, 2, 0);
  op_mpi_wait_all_grouped(1, args, 2, 0);

  const int setSize = _mesh->cells->size;
  DG_MAT_IND_TYPE *data_ptr = (DG_MAT_IND_TYPE *)malloc(setSize * sizeof(DG_MAT_IND_TYPE));
  hipMemcpy(data_ptr, glb_ind->data_d, setSize * sizeof(DG_MAT_IND_TYPE), hipMemcpyDeviceToHost);

  #pragma omp parallel for
  for(int i = 0; i < _mesh->cells->size; i++) {
    data_ptr[i] = global_ind + i * DG_NP_N1;
  }

  hipMemcpy(glb_ind->data_d, data_ptr, setSize * sizeof(DG_MAT_IND_TYPE), hipMemcpyHostToDevice);

  op_mpi_set_dirtybit_cuda(1, args);
  free(data_ptr);
}

void PoissonCoarseMatrix::setPETScMatrix() {
  if(!petscMatInit) {
    timer->startTimer("setPETScMatrix - Create Matrix");
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
    // Guess for the off diagonal preallocation at the moment
    MatMPIAIJSetPreallocation(pMat, DG_NP_N1 * (DG_NUM_FACES + 1), NULL, DG_NP_N1 * 2, NULL);
    #else
    #ifdef PETSC_COMPILED_WITH_HIP
    MatSetType(pMat, MATSEQAIJHIPSPARSE);
    #else
    MatSetType(pMat, MATSEQAIJ);
    #endif
    MatSeqAIJSetPreallocation(pMat, DG_NP_N1 * (DG_NUM_FACES + 1), NULL);
    #endif
    MatSetOption(pMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetOption(pMat, MAT_STRUCTURALLY_SYMMETRIC, PETSC_TRUE);
    MatSetOption(pMat, MAT_STRUCTURAL_SYMMETRY_ETERNAL, PETSC_TRUE);
    MatSetOption(pMat, MAT_SPD, PETSC_TRUE);
    MatSetOption(pMat, MAT_SPD_ETERNAL, PETSC_TRUE);
    timer->endTimer("setPETScMatrix - Create Matrix");
  } else {
    MatSetOption(pMat, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE);
    MatSetOption(pMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
    // MatSetOption(pMat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE);
    MatSetOption(pMat, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
  }

  timer->startTimer("setPETScMatrix - OP2 op1");
  // Add cubature OP to Poisson matrix
  op_arg args[] = {
    op_arg_dat(op1, -1, OP_ID, DG_NP_N1 * DG_NP_N1, DG_FP_STR, OP_READ),
    op_arg_dat(glb_ind, -1, OP_ID, 1, DG_MAT_IND_TYPE_STR, OP_READ)
  };
  op_mpi_halo_exchanges_grouped(_mesh->cells, 2, args, 2, 0);
  op_mpi_wait_all_grouped(2, args, 2, 0);
  timer->endTimer("setPETScMatrix - OP2 op1");

  timer->startTimer("setPETScMatrix - Copy op1 to host");
  #ifdef DG_OP2_SOA
  const int setSize = getSetSizeFromOpArg(&args[0]);
  #else
  const int setSize = _mesh->cells->size;
  #endif
  DG_FP *op1_data = (DG_FP *)malloc(DG_NP_N1 * DG_NP_N1 * setSize * sizeof(DG_FP));
  DG_MAT_IND_TYPE *glb = (DG_MAT_IND_TYPE *)malloc(setSize * sizeof(DG_MAT_IND_TYPE));
  hipMemcpy(op1_data, op1->data_d, setSize * DG_NP_N1 * DG_NP_N1 * sizeof(DG_FP), hipMemcpyDeviceToHost);
  hipMemcpy(glb, glb_ind->data_d, setSize * sizeof(DG_MAT_IND_TYPE), hipMemcpyDeviceToHost);
  timer->endTimer("setPETScMatrix - Copy op1 to host");
  op_mpi_set_dirtybit_cuda(2, args);

  #ifdef DG_COL_MAJ
  MatSetOption(pMat, MAT_ROW_ORIENTED, PETSC_FALSE);
  #else
  MatSetOption(pMat, MAT_ROW_ORIENTED, PETSC_TRUE);
  #endif

  timer->startTimer("setPETScMatrix - Set values op1");
  for(int i = 0; i < _mesh->cells->size; i++) {
    DG_MAT_IND_TYPE currentRow = glb[i];
    DG_MAT_IND_TYPE currentCol = glb[i];
    PetscInt idxm[DG_NP_N1], idxn[DG_NP_N1];
    for(DG_MAT_IND_TYPE n = 0; n < DG_NP_N1; n++) {
      idxm[n] = static_cast<PetscInt>(currentRow + n);
      idxn[n] = static_cast<PetscInt>(currentCol + n);
    }

    #ifdef DG_OP2_SOA
    DG_FP tmp_op1[DG_NP_N1 * DG_NP_N1];
    for(int n = 0; n < DG_NP_N1 * DG_NP_N1; n++) {
      tmp_op1[n] = op1_data[i + n * setSize];
    }
    MatSetValues(pMat, DG_NP_N1, idxm, DG_NP_N1, idxn, tmp_op1, INSERT_VALUES);
    #else
    MatSetValues(pMat, DG_NP_N1, idxm, DG_NP_N1, idxn, &op1_data[i * DG_NP_N1 * DG_NP_N1], INSERT_VALUES);
    #endif
  }
  timer->endTimer("setPETScMatrix - Set values op1");

  free(op1_data);
  free(glb);

  timer->startTimer("setPETScMatrix - OP2 op2");
  op_arg edge_args[] = {
    op_arg_dat(op2[0], -1, OP_ID, DG_NP_N1 * DG_NP_N1, DG_FP_STR, OP_READ),
    op_arg_dat(op2[1], -1, OP_ID, DG_NP_N1 * DG_NP_N1, DG_FP_STR, OP_READ),
    op_arg_dat(glb_indL, -1, OP_ID, 1, DG_MAT_IND_TYPE_STR, OP_READ),
    op_arg_dat(glb_indR, -1, OP_ID, 1, DG_MAT_IND_TYPE_STR, OP_READ)
  };
  op_mpi_halo_exchanges_grouped(_mesh->faces, 4, edge_args, 2, 0);
  op_mpi_wait_all_grouped(4, edge_args, 2, 0);
  timer->endTimer("setPETScMatrix - OP2 op2");

  timer->startTimer("setPETScMatrix - Copy op2 to host");
  #ifdef DG_OP2_SOA
  const int faces_set_size = getSetSizeFromOpArg(&edge_args[0]);
  #else
  const int faces_set_size = _mesh->faces->size;
  #endif
  DG_FP *op2L_data = (DG_FP *)malloc(DG_NP_N1 * DG_NP_N1 * faces_set_size * sizeof(DG_FP));
  DG_FP *op2R_data = (DG_FP *)malloc(DG_NP_N1 * DG_NP_N1 * faces_set_size * sizeof(DG_FP));
  DG_MAT_IND_TYPE *glb_l = (DG_MAT_IND_TYPE *)malloc(faces_set_size * sizeof(DG_MAT_IND_TYPE));
  DG_MAT_IND_TYPE *glb_r = (DG_MAT_IND_TYPE *)malloc(faces_set_size * sizeof(DG_MAT_IND_TYPE));

  hipMemcpy(op2L_data, op2[0]->data_d, DG_NP_N1 * DG_NP_N1 * faces_set_size * sizeof(DG_FP), hipMemcpyDeviceToHost);
  hipMemcpy(op2R_data, op2[1]->data_d, DG_NP_N1 * DG_NP_N1 * faces_set_size * sizeof(DG_FP), hipMemcpyDeviceToHost);
  hipMemcpy(glb_l, glb_indL->data_d, faces_set_size * sizeof(DG_MAT_IND_TYPE), hipMemcpyDeviceToHost);
  hipMemcpy(glb_r, glb_indR->data_d, faces_set_size * sizeof(DG_MAT_IND_TYPE), hipMemcpyDeviceToHost);
  timer->endTimer("setPETScMatrix - Copy op2 to host");

  op_mpi_set_dirtybit_cuda(4, edge_args);

  // Add Gauss OP and OPf to Poisson matrix
  timer->startTimer("setPETScMatrix - Set values op2");
  for(int i = 0; i < _mesh->faces->size; i++) {
    DG_MAT_IND_TYPE leftRow = glb_l[i];
    DG_MAT_IND_TYPE rightRow = glb_r[i];

    PetscInt idxl[DG_NP_N1], idxr[DG_NP_N1];
    for(DG_MAT_IND_TYPE n = 0; n < DG_NP_N1; n++) {
      idxl[n] = static_cast<PetscInt>(leftRow + n);
      idxr[n] = static_cast<PetscInt>(rightRow + n);
    }

    #ifdef DG_OP2_SOA
    DG_FP tmp_op2_l[DG_NP_N1 * DG_NP_N1], tmp_op2_r[DG_NP_N1 * DG_NP_N1];
    for(int n = 0; n < DG_NP_N1 * DG_NP_N1; n++) {
      tmp_op2_l[n] = op2L_data[i + n * faces_set_size];
      tmp_op2_r[n] = op2R_data[i + n * faces_set_size];
    }
    MatSetValues(pMat, DG_NP_N1, idxl, DG_NP_N1, idxr, tmp_op2_l, INSERT_VALUES);
    MatSetValues(pMat, DG_NP_N1, idxr, DG_NP_N1, idxl, tmp_op2_r, INSERT_VALUES);
    #else
    MatSetValues(pMat, DG_NP_N1, idxl, DG_NP_N1, idxr, &op2L_data[i * DG_NP_N1 * DG_NP_N1], INSERT_VALUES);
    MatSetValues(pMat, DG_NP_N1, idxr, DG_NP_N1, idxl, &op2R_data[i * DG_NP_N1 * DG_NP_N1], INSERT_VALUES);
    #endif
  }
  timer->endTimer("setPETScMatrix - Set values op2");

  free(op2L_data);
  free(op2R_data);
  free(glb_l);
  free(glb_r);

  timer->startTimer("setPETScMatrix - Assembly");
  MatAssemblyBegin(pMat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(pMat, MAT_FINAL_ASSEMBLY);
  timer->endTimer("setPETScMatrix - Assembly");
}

#ifdef DG_MPI
#include "mpi.h"
#endif

#include "dg_mesh/dg_mesh_3d.h"

#ifdef INS_BUILD_WITH_HYPRE
void PoissonCoarseMatrix::setHYPREMatrix() {
  dg_abort("PoissonCoarseMatrix::setHYPREMatrix not implemented for HIP yet");
  /*
  int global_size = getUnknowns();
  int local_size = getUnknowns();
  #ifdef DG_MPI
  global_size = global_sum(global_size);
  #endif
  // Keep track of how many non-zero entries locally
  // int nnz = 0;
  const int cell_set_size = _mesh->cells->size;
  const int faces_set_size = _mesh->faces->size + _mesh->faces->exec_size + _mesh->faces->nonexec_size;
  int nnz = cell_set_size * DG_NP_N1 * DG_NP_N1 + faces_set_size * DG_NP_N1 * DG_NP_N1 * 2;

  float *data_buf_ptr_h = (float *)malloc(nnz * sizeof(float));
  int *col_buf_ptr_h = (int *)malloc(nnz * sizeof(int));
  int *row_num_ptr_h = (int *)malloc(local_size * sizeof(int));
  int *num_col_ptr_h = (int *)malloc(local_size * sizeof(int));

  // Exchange halos
  op_arg args[] = {
    op_arg_dat(glb_indL, -1, OP_ID, glb_indL->dim, "int", OP_RW),
    op_arg_dat(glb_indR, -1, OP_ID, glb_indR->dim, "int", OP_RW)
  };
  op_mpi_halo_exchanges_grouped(_mesh->faces, 2, args, 2, 1);
  op_mpi_wait_all_grouped(2, args, 2, 1);
  cudaDeviceSynchronize();

  // Get data from OP2
  DG_FP *op1_data = getOP2PtrHostHE(op1, OP_READ);
  DG_FP *op2L_data = getOP2PtrHostHE(op2[0], OP_READ);
  DG_FP *op2R_data = getOP2PtrHostHE(op2[1], OP_READ);
  int *glb   = (int *)malloc(cell_set_size * sizeof(int));
  hipMemcpy(glb, glb_ind->data_d, cell_set_size * sizeof(int), hipMemcpyDeviceToHost);
  int *glb_l = (int *)malloc(faces_set_size * sizeof(int));
  int *glb_r = (int *)malloc(faces_set_size * sizeof(int));
  hipMemcpy(glb_l, glb_indL->data_d, faces_set_size * sizeof(int), hipMemcpyDeviceToHost);
  hipMemcpy(glb_r, glb_indR->data_d, faces_set_size * sizeof(int), hipMemcpyDeviceToHost);

  if(!hypre_mat_init) {
    const int ilower = glb[0];
    const int iupper = glb[0] + local_size - 1;
    // printf("ilower: %d iupper: %d\n", ilower, iupper);
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &hypre_mat);
    HYPRE_IJMatrixSetObjectType(hypre_mat, HYPRE_PARCSR);
    // HYPRE_IJMatrixInitialize(hypre_mat);
    hypre_mat_init = true;
  }

  HYPRE_IJMatrixInitialize(hypre_mat);

  std::map<int,std::vector<std::pair<int,DG_FP*>>> mat_buffer;

  timer->startTimer("HYPRE - map0");
  for(int c = 0; c < cell_set_size; c++) {
    // Add diagonal block to buffer
    int diag_base_col = glb[c];
    std::vector<std::pair<int,DG_FP*>> row_buf;
    row_buf.push_back({diag_base_col, op1_data + c * DG_NP_N1 * DG_NP_N1});
    mat_buffer.insert({diag_base_col, row_buf});
  }
  timer->endTimer("HYPRE - map0");

  timer->startTimer("HYPRE - map1");
  for(int k = 0; k < faces_set_size; k++) {
    if(glb_l[k] >= glb[0] && glb_l[k] < glb[0] + local_size) {
      int base_col = glb_r[k];
      DG_FP *face_data_ptr = op2L_data + k * DG_NP_N1 * DG_NP_N1;
      std::vector<std::pair<int,DG_FP*>> &row_buf = mat_buffer.at(glb_l[k]);
      row_buf.push_back({base_col, face_data_ptr});
    }
  }
  timer->endTimer("HYPRE - map1");

  timer->startTimer("HYPRE - map2");
  for(int k = 0; k < faces_set_size; k++) {
    if(glb_r[k] >= glb[0] && glb_r[k] < glb[0] + local_size) {
      int base_col = glb_l[k];
      DG_FP *face_data_ptr = op2R_data + k * DG_NP_N1 * DG_NP_N1;
      std::vector<std::pair<int,DG_FP*>> &row_buf = mat_buffer.at(glb_r[k]);
      row_buf.push_back({base_col, face_data_ptr});
    }
  }
  timer->endTimer("HYPRE - map2");
  timer->endTimer("HYPRE - map");

  timer->startTimer("HYPRE - convert format");
  int current_nnz = 0;
  int current_row = 0;
  for(auto it = mat_buffer.begin(); it != mat_buffer.end(); it++) {
    std::sort(it->second.begin(), it->second.end());

    for(int i = 0; i < DG_NP_N1; i++) {
      row_num_ptr_h[current_row] = it->first + i;
      int num_this_col = 0;
      for(int elem = 0; elem < it->second.size(); elem++) {
        DG_FP *data_ptr = it->second[elem].second;
        int base_col_ind = it->second[elem].first;
        for(int j = 0; j < DG_NP_N1; j++) {
          col_buf_ptr_h[current_nnz] = base_col_ind + j;
          data_buf_ptr_h[current_nnz] = data_ptr[i + j * DG_NP_N1];
          num_this_col++;
          current_nnz++;
        }
      }
      num_col_ptr_h[current_row] = num_this_col;
      current_row++;
    }
  }
  timer->endTimer("HYPRE - convert format");

  timer->startTimer("HYPRE - transfer 2");
  float *data_buf_ptr;
  cudaMalloc(&data_buf_ptr, current_nnz * sizeof(float));
  int *col_buf_ptr;
  cudaMalloc(&col_buf_ptr, current_nnz * sizeof(int));
  int *row_num_ptr;
  cudaMalloc(&row_num_ptr, local_size * sizeof(int));
  int *num_col_ptr;
  cudaMalloc(&num_col_ptr, local_size * sizeof(int));

  hipMemcpy(row_num_ptr, row_num_ptr_h, local_size * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(num_col_ptr, num_col_ptr_h, local_size * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(col_buf_ptr, col_buf_ptr_h, current_nnz * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(data_buf_ptr, data_buf_ptr_h, current_nnz * sizeof(float), hipMemcpyHostToDevice);
  timer->endTimer("HYPRE - transfer 2");

  timer->startTimer("HYPRE - construct");
  // HYPRE_IJMatrixSetRowSizes(hypre_mat, num_col_ptr);
  HYPRE_IJMatrixSetValues(hypre_mat, local_size, num_col_ptr, row_num_ptr, col_buf_ptr, data_buf_ptr);

  cudaFree(data_buf_ptr);
  cudaFree(col_buf_ptr);
  cudaFree(row_num_ptr);
  cudaFree(num_col_ptr);
  free(data_buf_ptr_h);
  free(col_buf_ptr_h);
  free(row_num_ptr_h);
  free(num_col_ptr_h);

  releaseOP2PtrHostHE(op1, OP_READ, op1_data);
  releaseOP2PtrHostHE(op2[0], OP_READ, op2L_data);
  releaseOP2PtrHostHE(op2[1], OP_READ, op2R_data);
  free(glb);
  free(glb_l);
  free(glb_r);

  HYPRE_IJMatrixAssemble(hypre_mat);
  timer->endTimer("HYPRE - construct");
  */
}
#endif
