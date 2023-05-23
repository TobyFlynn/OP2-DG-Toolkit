#include "dg_dat_pool.h"

#include <string>

#ifdef DG_MPI
#include "mpi.h"
#endif

DGDatPool::DGDatPool(DGMesh *m) {
  mesh = m;
}

DGDatPool::~DGDatPool() {
  for(int i = 0; i < cell_dats.size(); i++) {
    op_free_dat_temp(cell_dats[i].dat);
  }

  for(int i = 0; i < face_dats.size(); i++) {
    op_free_dat_temp(face_dats[i].dat);
  }
}

DGTempDat DGDatPool::requestTempDatCells(const int dim) {
  for(int i = 0; i < cell_dats.size(); i++) {
    if(cell_dats[i].dat->dim == dim && !cell_dats[i].in_use) {
      cell_dats[i].in_use = true;
      DGTempDat tmp;
      tmp.dat = cell_dats[i].dat;
      tmp.ind = i;
      return tmp;
    }
  }

  std::string tmp_dat_name = "dg_dat_pool_cells_" + std::to_string(cell_dats.size());
  op_dat tmp_dat = op_decl_dat_temp(mesh->cells, dim, DG_FP_STR, (DG_FP *)NULL, tmp_dat_name.c_str());

  cell_dats.push_back({tmp_dat, true});

  DGTempDat tmp;
  tmp.dat = tmp_dat;
  tmp.ind = cell_dats.size() - 1;
  return tmp;
}

void DGDatPool::releaseTempDatCells(DGTempDat tempDat) {
  cell_dats[tempDat.ind].in_use = false;
}

DGTempDat DGDatPool::requestTempDatFaces(const int dim) {
  for(int i = 0; i < face_dats.size(); i++) {
    if(face_dats[i].dat->dim == dim && !face_dats[i].in_use) {
      face_dats[i].in_use = true;
      DGTempDat tmp;
      tmp.dat = face_dats[i].dat;
      tmp.ind = i;
      return tmp;
    }
  }

  std::string tmp_dat_name = "dg_dat_pool_faces_" + std::to_string(face_dats.size());
  op_dat tmp_dat = op_decl_dat_temp(mesh->faces, dim, DG_FP_STR, (DG_FP *)NULL, tmp_dat_name.c_str());

  face_dats.push_back({tmp_dat, true});

  DGTempDat tmp;
  tmp.dat = tmp_dat;
  tmp.ind = face_dats.size() - 1;
  return tmp;
}

void DGDatPool::releaseTempDatFaces(DGTempDat tempDat) {
  face_dats[tempDat.ind].in_use = false;
}

void DGDatPool::report() {
  int dim_cells = 0;
  int dim_faces = 0;

  for(int i = 0; i < cell_dats.size(); i++) {
    dim_cells += cell_dats[i].dat->dim;
  }

  for(int i = 0; i < face_dats.size(); i++) {
    dim_faces += face_dats[i].dat->dim;
  }

  int num_cells = 0;
  int num_faces = 0;

  #ifdef DG_MPI
  MPI_Allreduce(&mesh->cells->size, &num_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mesh->faces->size, &num_faces, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  #else
  num_cells = mesh->cells->size;
  num_faces = mesh->faces->size;
  #endif

  double gb_used = (dim_cells * num_cells + dim_faces * num_cells) * sizeof(DG_FP) / 1e9;
  op_printf("Amount of memory used by DG Dat Pool is: %g GB\n", gb_used);
}
