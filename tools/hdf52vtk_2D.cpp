#include <vtkSmartPointer.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellIterator.h>
#include <vtkIdList.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkDataSetAttributes.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "highfive/H5File.hpp"

#include "dg_compiler_defs.h"

using HighFive::File;

void add_2d_vec_solution(File *file, const std::string &nameX, const std::string &nameY,
                         const std::string &nameVec, const int numCells,
                         vtkUnstructuredGrid *vtkGrid) {
  if(file->exist(nameX) && file->exist(nameY)) {
    std::vector<std::vector<double>> u_vec, v_vec;
    file->getDataSet(nameX).read(u_vec);
    file->getDataSet(nameY).read(v_vec);
    vtkNew<vtkDoubleArray> sol_vector;
    sol_vector->SetName(nameVec.c_str());
    sol_vector->SetNumberOfComponents(2);
    sol_vector->SetNumberOfTuples(numCells * DG_NP);
    for(int i = 0; i < numCells; i++) {
      for(int j = 0; j < DG_NP; j++) {
        sol_vector->SetTuple2(i * DG_NP + j, u_vec[i][j], v_vec[i][j]);
      }
    }
    vtkGrid->GetPointData()->AddArray(sol_vector);
  }
}

void add_1d_vec_solution(File *file, const std::string &name, const std::string &nameVec,
                         const int numCells, vtkUnstructuredGrid *vtkGrid) {
  if(file->exist(name)) {
    std::vector<std::vector<double>> u_vec;
    file->getDataSet(name).read(u_vec);
    vtkNew<vtkDoubleArray> sol_vector;
    sol_vector->SetName(nameVec.c_str());
    sol_vector->SetNumberOfComponents(1);
    sol_vector->SetNumberOfTuples(numCells * DG_NP);
    for(int i = 0; i < numCells; i++) {
      for(int j = 0; j < DG_NP; j++) {
        sol_vector->SetTuple1(i * DG_NP + j, u_vec[i][j]);
      }
    }
    vtkGrid->GetPointData()->AddArray(sol_vector);
  }
}

int main(int argc, char **argv) {
  for(int arg = 1; arg < argc; arg++) {
    std::string filename = argv[arg];

    File file(filename, File::ReadOnly);
    std::vector<std::vector<double>> x_vec, y_vec;
    file.getDataSet("x").read(x_vec);
    file.getDataSet("y").read(y_vec);

    const int numCells = x_vec.size();

    vtkNew<vtkUnstructuredGrid> vtkGrid;
    vtkNew<vtkPoints> vtkPoints;
    // TODO is this right? shouldn't it be x_vec.size() * DG_NP?
    vtkPoints->Allocate(x_vec.size());

    // Set points
    for(int cell = 0; cell < numCells; cell++) {
      for(int i = 0; i < DG_NP; i++) {
        vtkPoints->InsertNextPoint(x_vec[cell][i], y_vec[cell][i], 0.0);
      }
    }
    vtkGrid->SetPoints(vtkPoints);

    #if DG_ORDER == 1
    const int numSubCells = 1;
    const vtkIdType subCellMap[1][3] = {
      {0, 1, 2}
    };
    #elif DG_ORDER == 2
    const int numSubCells = 4;
    const vtkIdType subCellMap[4][3] = {
      {0, 1, 3},
      {1, 3, 4},
      {1, 2, 4},
      {3, 4, 5}
    };
    #elif DG_ORDER == 3
    const int numSubCells = 9;
    const vtkIdType subCellMap[9][3] = {
      {0, 1, 4},
      {1, 4, 5},
      {1, 2, 5},
      {2, 5, 6},
      {2, 3, 6},
      {4, 5, 7},
      {5, 7, 8},
      {5, 6, 8},
      {7, 8, 9}
    };
    #elif DG_ORDER == 4
    const int numSubCells = 16;
    const vtkIdType subCellMap[16][3] = {
      {0, 1, 5},
      {1, 5, 6},
      {1, 2, 6},
      {2, 6, 7},
      {2, 3, 7},
      {3, 7, 8},
      {3, 4, 8},
      {5, 6, 9},
      {6, 9, 10},
      {6, 7, 10},
      {7, 10, 11},
      {7, 8, 11},
      {9, 10, 12},
      {10, 12, 13},
      {10, 11, 13},
      {12, 13, 14}
    };
    #endif

    vtkGrid->Allocate(numCells * numSubCells);
    for(int cell = 0; cell < numCells; cell++) {
      for(int i = 0; i < numSubCells; i++) {
        const int basePtInd = cell * DG_NP;
        vtkIdType ptIds[] = {subCellMap[i][0] + basePtInd, subCellMap[i][1] + basePtInd, subCellMap[i][2] + basePtInd};
        vtkGrid->InsertNextCell(VTK_TRIANGLE, 3, ptIds);
      }
    }

    add_2d_vec_solution(&file, "ins_solver_vel00", "ins_solver_vel01", "velocity0", numCells, vtkGrid);
    add_2d_vec_solution(&file, "ins_solver_vel10", "ins_solver_vel11", "velocity1", numCells, vtkGrid);
    add_2d_vec_solution(&file, "ins_solver_velT0", "ins_solver_velT1", "velocityT", numCells, vtkGrid);
    add_2d_vec_solution(&file, "ins_solver_velTT0", "ins_solver_velTT1", "velocityTT", numCells, vtkGrid);
    add_1d_vec_solution(&file, "ins_solver_pr", "pressure", numCells, vtkGrid);
    add_1d_vec_solution(&file, "ins_solver_temperature", "temperature", numCells, vtkGrid);
    add_1d_vec_solution(&file, "ins_solver_rho", "rho", numCells, vtkGrid);

    add_2d_vec_solution(&file, "mp_ins_solver_vel00", "mp_ins_solver_vel01", "velocity0", numCells, vtkGrid);
    add_2d_vec_solution(&file, "mp_ins_solver_vel10", "mp_ins_solver_vel11", "velocity1", numCells, vtkGrid);
    add_2d_vec_solution(&file, "mp_ins_solver_velT0", "mp_ins_solver_velT1", "velocityT", numCells, vtkGrid);
    add_2d_vec_solution(&file, "mp_ins_solver_velTT0", "mp_ins_solver_velTT1", "velocityTT", numCells, vtkGrid);
    add_1d_vec_solution(&file, "mp_ins_solver_pr", "pressure", numCells, vtkGrid);
    add_1d_vec_solution(&file, "mp_ins_solver_rho", "rho", numCells, vtkGrid);
    add_1d_vec_solution(&file, "mp_ins_solver_mu", "mu", numCells, vtkGrid);
    add_1d_vec_solution(&file, "ls_s", "level_set", numCells, vtkGrid);

    add_2d_vec_solution(&file, "ins_solver_st00", "ins_solver_st01", "SurfTen0", numCells, vtkGrid);
    add_2d_vec_solution(&file, "ins_solver_st10", "ins_solver_st11", "SurfTen1", numCells, vtkGrid);
    add_2d_vec_solution(&file, "ins_solver_n00", "ins_solver_n01", "N0", numCells, vtkGrid);
    add_2d_vec_solution(&file, "ins_solver_n10", "ins_solver_n11", "N1", numCells, vtkGrid);

    add_1d_vec_solution(&file, "Q0", "Q0", numCells, vtkGrid);
    add_1d_vec_solution(&file, "Q1", "Q1", numCells, vtkGrid);
    add_1d_vec_solution(&file, "Q2", "Q2", numCells, vtkGrid);
    add_1d_vec_solution(&file, "Q3", "Q3", numCells, vtkGrid);

    vtkNew<vtkUnstructuredGridWriter> writer;
    std::string outfile = filename.substr(0,filename.size() - 3) + ".vtk";
    writer->SetFileName(outfile.c_str());
    writer->SetInputData(vtkGrid);
    writer->Write();
  }
}
