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

int main(int argc, char **argv) {
  for(int arg = 1; arg < argc; arg++) {
    std::string filename = argv[arg];

    File file(filename, File::ReadOnly);
    std::vector<std::vector<double>> x_vec, y_vec, z_vec;
    file.getDataSet("x").read(x_vec);
    file.getDataSet("y").read(y_vec);
    file.getDataSet("z").read(z_vec);

    const int numCells = x_vec.size();

    vtkNew<vtkUnstructuredGrid> vtkGrid;
    vtkNew<vtkPoints> vtkPoints;
    vtkPoints->Allocate(x_vec.size());

    // Set points
    for(int cell = 0; cell < numCells; cell++) {
      for(int i = 0; i < DG_NP; i++) {
        vtkPoints->InsertNextPoint(x_vec[cell][i], y_vec[cell][i], z_vec[cell][i]);
      }
    }
    vtkGrid->SetPoints(vtkPoints);

    // Set cells (subdivide high order cells into smaller cells)
    #if DG_ORDER == 1
    const int numSubCells = 1;
    const vtkIdType subCellMap[1][4] = {
      {1, 2, 0, 3}
    };
    #elif DG_ORDER == 2
    const int numSubCells = 8;
    const vtkIdType subCellMap[8][4] = {
      {1, 3, 0, 6},
      {4, 3, 1, 6},
      {2, 4, 1, 7},
      {4, 6, 1, 7},
      {6, 7, 4, 8},
      {5, 3, 4, 8},
      {3, 6, 4, 8},
      {7, 8, 6, 9}
    };
    #elif DG_ORDER == 3
    const int numSubCells = 29;
    const vtkIdType subCellMap[29][4] = {
      {1, 4, 0, 10},
      {11, 12, 6, 14},
      {13, 11, 10, 14},
      {8, 13, 5, 14},
      {8, 5, 6, 14},
      {6, 5, 2, 11},
      {2, 5, 1, 11},
      {11, 13, 10, 16},
      {3, 6, 2, 12},
      {6, 11, 2, 12},
      {7, 4, 5, 13},
      {5, 11, 6, 14},
      {8, 7, 5, 13},
      {13, 14, 8, 15},
      {10, 11, 1, 14},
      {1, 11, 5, 14},
      {5, 13, 4, 14},
      {13, 10, 4, 14},
      {5, 4, 1, 14},
      {4, 10, 1, 14},
      {9, 7, 8, 15},
      {7, 13, 8, 15},
      {12, 14, 11, 17},
      {14, 13, 11, 17},
      {13, 16, 11, 17},
      {16, 17, 13, 18},
      {14, 15, 13, 18},
      {13, 17, 14, 18},
      {18, 16, 17, 19}
    };
    #elif DG_ORDER == 4
    const int numSubCells = 73;
    const vtkIdType subCellMap[73][4] = {
      {1, 5, 0, 15},
      {6, 5, 1, 15},
      {9, 5, 6, 19},
      {18, 21, 17, 27},
      {20, 19, 16, 25},
      {8, 17, 3, 18},
      {16, 19, 15, 25},
      {2, 6, 1, 16},
      {8, 7, 3, 17},
      {3, 7, 2, 17},
      {11, 10, 6, 20},
      {4, 8, 3, 18},
      {6, 15, 1, 16},
      {15, 16, 6, 19},
      {11, 6, 7, 20},
      {10, 20, 11, 23},
      {5, 15, 6, 19},
      {22, 23, 13, 24},
      {11, 20, 7, 21},
      {19, 16, 6, 20},
      {13, 12, 10, 22},
      {22, 10, 13, 23},
      {9, 19, 6, 20},
      {17, 18, 8, 21},
      {6, 16, 2, 20},
      {11, 7, 8, 21},
      {7, 17, 8, 21},
      {2, 17, 7, 21},
      {7, 6, 2, 21},
      {6, 20, 2, 21},
      {20, 6, 7, 21},
      {16, 17, 2, 21},
      {2, 20, 16, 21},
      {20, 17, 16, 21},
      {12, 9, 10, 22},
      {13, 10, 11, 23},
      {20, 21, 11, 23},
      {10, 22, 9, 23},
      {10, 9, 6, 23},
      {9, 20, 6, 23},
      {6, 20, 10, 23},
      {20, 9, 19, 23},
      {9, 22, 19, 23},
      {19, 22, 20, 23},
      {14, 12, 13, 24},
      {12, 22, 13, 24},
      {21, 23, 20, 29},
      {28, 25, 26, 31},
      {17, 20, 16, 27},
      {20, 26, 16, 27},
      {21, 20, 17, 27},
      {23, 24, 22, 30},
      {20, 27, 21, 29},
      {28, 29, 20, 30},
      {26, 27, 20, 29},
      {28, 26, 25, 29},
      {20, 25, 16, 29},
      {25, 26, 16, 29},
      {16, 26, 20, 29},
      {19, 25, 20, 29},
      {20, 28, 19, 29},
      {28, 25, 19, 29},
      {20, 29, 23, 30},
      {20, 22, 19, 30},
      {19, 28, 20, 30},
      {23, 22, 20, 30},
      {29, 26, 27, 32},
      {29, 28, 26, 32},
      {28, 31, 26, 32},
      {32, 28, 31, 33},
      {30, 28, 29, 33},
      {28, 32, 29, 33},
      {32, 33, 31, 34}
    };
    #endif

    vtkGrid->Allocate(numCells * numSubCells);
    for(int cell = 0; cell < numCells; cell++) {
      for(int i = 0; i < numSubCells; i++) {
        const int basePtInd = cell * DG_NP;
        vtkIdType ptIds[] = {subCellMap[i][0] + basePtInd, subCellMap[i][1] + basePtInd, subCellMap[i][2] + basePtInd, subCellMap[i][3] + basePtInd};
        vtkGrid->InsertNextCell(VTK_TETRA, 4, ptIds);
      }
    }

    std::vector<std::vector<double>> u_vec, v_vec, w_vec;
    file.getDataSet("ins_solver_vel00").read(u_vec);
    file.getDataSet("ins_solver_vel01").read(v_vec);
    file.getDataSet("ins_solver_vel02").read(w_vec);

    vtkNew<vtkDoubleArray> vel_vector;
    vel_vector->SetName("velocity");
    vel_vector->SetNumberOfComponents(3);
    vel_vector->SetNumberOfTuples(numCells * DG_NP);
    for(int i = 0; i < numCells; i++) {
      for(int j = 0; j < DG_NP; j++) {
        vel_vector->SetTuple3(i * DG_NP + j, u_vec[i][j], v_vec[i][j], w_vec[i][j]);
      }
    }
    vtkGrid->GetPointData()->AddArray(vel_vector);

    std::vector<std::vector<double>> u1_vec, v1_vec, w1_vec;
    file.getDataSet("ins_solver_vel10").read(u1_vec);
    file.getDataSet("ins_solver_vel11").read(v1_vec);
    file.getDataSet("ins_solver_vel12").read(w1_vec);

    vtkNew<vtkDoubleArray> vel1_vector;
    vel1_vector->SetName("velocity1");
    vel1_vector->SetNumberOfComponents(3);
    vel1_vector->SetNumberOfTuples(numCells * DG_NP);
    for(int i = 0; i < numCells; i++) {
      for(int j = 0; j < DG_NP; j++) {
        vel1_vector->SetTuple3(i * DG_NP + j, u1_vec[i][j], v1_vec[i][j], w1_vec[i][j]);
      }
    }
    vtkGrid->GetPointData()->AddArray(vel1_vector);

    std::vector<std::vector<double>> uT_vec, vT_vec, wT_vec;
    file.getDataSet("ins_solver_velT0").read(uT_vec);
    file.getDataSet("ins_solver_velT1").read(vT_vec);
    file.getDataSet("ins_solver_velT2").read(wT_vec);

    vtkNew<vtkDoubleArray> velT_vector;
    velT_vector->SetName("velocityT");
    velT_vector->SetNumberOfComponents(3);
    velT_vector->SetNumberOfTuples(numCells * DG_NP);
    for(int i = 0; i < numCells; i++) {
      for(int j = 0; j < DG_NP; j++) {
        velT_vector->SetTuple3(i * DG_NP + j, uT_vec[i][j], vT_vec[i][j], wT_vec[i][j]);
      }
    }
    vtkGrid->GetPointData()->AddArray(velT_vector);

    std::vector<std::vector<double>> uTT_vec, vTT_vec, wTT_vec;
    file.getDataSet("ins_solver_velTT0").read(uTT_vec);
    file.getDataSet("ins_solver_velTT1").read(vTT_vec);
    file.getDataSet("ins_solver_velTT2").read(wTT_vec);
    vtkNew<vtkDoubleArray> velTT_vector;
    velTT_vector->SetName("velocityTT");
    velTT_vector->SetNumberOfComponents(3);
    velTT_vector->SetNumberOfTuples(numCells * DG_NP);
    for(int i = 0; i < numCells; i++) {
      for(int j = 0; j < DG_NP; j++) {
        velTT_vector->SetTuple3(i * DG_NP + j, uTT_vec[i][j], vTT_vec[i][j], wTT_vec[i][j]);
      }
    }
    vtkGrid->GetPointData()->AddArray(velTT_vector);

    std::vector<std::vector<double>> pr_vec;
    file.getDataSet("ins_solver_pr").read(pr_vec);
    vtkNew<vtkDoubleArray> pr_vector;
    pr_vector->SetName("Pressure");
    pr_vector->SetNumberOfComponents(1);
    pr_vector->SetNumberOfTuples(numCells * DG_NP);
    for(int i = 0; i < numCells; i++) {
      for(int j = 0; j < DG_NP; j++) {
        pr_vector->SetTuple1(i * DG_NP + j, pr_vec[i][j]);
      }
    }
    vtkGrid->GetPointData()->AddArray(pr_vector);

    if(file.exist("ins_solver_rho")) {
      std::vector<std::vector<double>> rho_vec;
      file.getDataSet("ins_solver_rho").read(rho_vec);
      vtkNew<vtkDoubleArray> rho_vector;
      rho_vector->SetName("Rho");
      rho_vector->SetNumberOfComponents(1);
      rho_vector->SetNumberOfTuples(numCells * DG_NP);
      for(int i = 0; i < numCells; i++) {
        for(int j = 0; j < DG_NP; j++) {
          rho_vector->SetTuple1(i * DG_NP + j, rho_vec[i][j]);
        }
      }
      vtkGrid->GetPointData()->AddArray(rho_vector);
    }

    if(file.exist("ins_solver_mu")) {
      std::vector<std::vector<double>> mu_vec;
      file.getDataSet("ins_solver_mu").read(mu_vec);
      vtkNew<vtkDoubleArray> mu_vector;
      mu_vector->SetName("Mu");
      mu_vector->SetNumberOfComponents(1);
      mu_vector->SetNumberOfTuples(numCells * DG_NP);
      for(int i = 0; i < numCells; i++) {
        for(int j = 0; j < DG_NP; j++) {
          mu_vector->SetTuple1(i * DG_NP + j, mu_vec[i][j]);
        }
      }
      vtkGrid->GetPointData()->AddArray(mu_vector);
    }

    if(file.exist("ls_solver_s")) {
      std::vector<std::vector<double>> s_vec;
      file.getDataSet("ls_solver_s").read(s_vec);
      vtkNew<vtkDoubleArray> s_vector;
      s_vector->SetName("Surface");
      s_vector->SetNumberOfComponents(1);
      s_vector->SetNumberOfTuples(numCells * DG_NP);
      for(int i = 0; i < numCells; i++) {
        for(int j = 0; j < DG_NP; j++) {
          s_vector->SetTuple1(i * DG_NP + j, s_vec[i][j]);
        }
      }
      vtkGrid->GetPointData()->AddArray(s_vector);
    }

    vtkNew<vtkUnstructuredGridWriter> writer;
    std::string outfile = filename.substr(0,filename.size() - 3) + ".vtk";
    writer->SetFileName(outfile.c_str());
    writer->SetInputData(vtkGrid);
    writer->Write();
  }
}