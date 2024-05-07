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
#include <map>

#include "highfive/H5File.hpp"
#include "CDT.h"

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

std::vector<std::vector<int>> get_sub_cell_map(const std::vector<double> &x, const std::vector<double> &y);

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
    vtkPoints->Allocate(x_vec.size() * DG_NP);

    // Set points
    for(int cell = 0; cell < numCells; cell++) {
      for(int i = 0; i < DG_NP; i++) {
        vtkPoints->InsertNextPoint(x_vec[cell][i], y_vec[cell][i], 0.0);
      }
    }
    vtkGrid->SetPoints(vtkPoints);

    std::vector<std::vector<int>> subCellMap = get_sub_cell_map(x_vec[0], y_vec[0]);

    vtkGrid->Allocate(numCells * subCellMap.size());
    for(int cell = 0; cell < numCells; cell++) {
      for(int i = 0; i < subCellMap.size(); i++) {
        const int basePtInd = cell * DG_NP;
        vtkIdType ptIds[] = {subCellMap[i][0] + basePtInd, subCellMap[i][1] + basePtInd, subCellMap[i][2] + basePtInd};
        vtkGrid->InsertNextCell(VTK_TRIANGLE, 3, ptIds);
      }
    }

    add_2d_vec_solution(&file, "ins_solver_vel00", "ins_solver_vel01", "velocity", numCells, vtkGrid);
    add_2d_vec_solution(&file, "ins_solver_vel10", "ins_solver_vel11", "velocity", numCells, vtkGrid);
    add_2d_vec_solution(&file, "ins_solver_velT0", "ins_solver_velT1", "velocityT", numCells, vtkGrid);
    add_2d_vec_solution(&file, "ins_solver_velTT0", "ins_solver_velTT1", "velocityTT", numCells, vtkGrid);
    add_1d_vec_solution(&file, "ins_solver_pr", "pressure", numCells, vtkGrid);
    add_1d_vec_solution(&file, "ins_solver_temperature", "temperature", numCells, vtkGrid);
    add_1d_vec_solution(&file, "ins_solver_rho", "rho", numCells, vtkGrid);
    add_1d_vec_solution(&file, "ins_solver_shock_cap_art_vis", "art_vis", numCells, vtkGrid);
    add_1d_vec_solution(&file, "ins_solver_mu", "mu", numCells, vtkGrid);
    add_1d_vec_solution(&file, "ls_s", "level_set", numCells, vtkGrid);
    add_1d_vec_solution(&file, "ls_kink", "kink", numCells, vtkGrid);
    add_1d_vec_solution(&file, "ins_solver_curvature", "curvature", numCells, vtkGrid);
    add_1d_vec_solution(&file, "err", "error", numCells, vtkGrid);
    add_1d_vec_solution(&file, "ins_solver_p_star", "p_star", numCells, vtkGrid);

    add_2d_vec_solution(&file, "ins_solver_st00", "ins_solver_st01", "SurfTen", numCells, vtkGrid);
    add_2d_vec_solution(&file, "ins_solver_st10", "ins_solver_st11", "SurfTen", numCells, vtkGrid);
    add_2d_vec_solution(&file, "ins_solver_n00", "ins_solver_n01", "NonLinear", numCells, vtkGrid);
    add_2d_vec_solution(&file, "ins_solver_n10", "ins_solver_n11", "NonLinear", numCells, vtkGrid);

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

struct Point {
  double x;
  double y;
};

bool double_eq(const double d0, const double d1) {
  return fabs(d0 - d1) < 1e-8;
}

std::vector<std::vector<int>> get_sub_cell_map_CDT(const std::vector<double> &x, const std::vector<double> &y) {
  std::vector<Point> pts;
  for(int i = 0; i < x.size(); i++) {
    pts.push_back({x[i], y[i]});
  }

  CDT::Triangulation<double> cdt;
  cdt.insertVertices(pts.begin(), pts.end(), [](const Point& p){ return p.x; }, [](const Point& p){ return p.y; });
  cdt.eraseSuperTriangle();
  auto triangles = cdt.triangles;
  auto vertices = cdt.vertices;

  std::map<int,int> vertices_map;
  for(int i = 0; i < pts.size(); i++) {
    for(int j = 0; j < pts.size(); j++) {
      if(double_eq(vertices[i].x, pts[j].x) && double_eq(vertices[i].y, pts[j].y)) {
        vertices_map.insert({i, j});
        break;
      }
    }
  }

  std::vector<std::vector<int>> sub_cell_map;
  for(const auto &tri : triangles) {
    sub_cell_map.push_back({
      vertices_map.at(tri.vertices[0]),
      vertices_map.at(tri.vertices[1]),
      vertices_map.at(tri.vertices[2])
    });
  }

  return sub_cell_map;
}

std::vector<std::vector<int>> get_sub_cell_map(const std::vector<double> &x, const std::vector<double> &y) {
  return get_sub_cell_map_CDT(x, y);

  std::vector<std::vector<int>> sub_cell_map;

  #if DG_ORDER == 1
  sub_cell_map.push_back({0, 1, 2});
  #elif DG_ORDER == 2
  sub_cell_map.push_back({0, 1, 3});
  sub_cell_map.push_back({1, 3, 4});
  sub_cell_map.push_back({1, 2, 4});
  sub_cell_map.push_back({3, 4, 5});
  #elif DG_ORDER == 3
  sub_cell_map.push_back({0, 1, 4});
  sub_cell_map.push_back({1, 4, 5});
  sub_cell_map.push_back({1, 2, 5});
  sub_cell_map.push_back({2, 5, 6});
  sub_cell_map.push_back({2, 3, 6});
  sub_cell_map.push_back({4, 5, 7});
  sub_cell_map.push_back({5, 7, 8});
  sub_cell_map.push_back({5, 6, 8});
  sub_cell_map.push_back({7, 8, 9});
  #elif DG_ORDER == 4
  sub_cell_map.push_back({0, 1, 5});
  sub_cell_map.push_back({1, 5, 6});
  sub_cell_map.push_back({1, 2, 6});
  sub_cell_map.push_back({2, 6, 7});
  sub_cell_map.push_back({2, 3, 7});
  sub_cell_map.push_back({3, 7, 8});
  sub_cell_map.push_back({3, 4, 8});
  sub_cell_map.push_back({5, 6, 9});
  sub_cell_map.push_back({6, 9, 10});
  sub_cell_map.push_back({6, 7, 10});
  sub_cell_map.push_back({7, 10, 11});
  sub_cell_map.push_back({7, 8, 11});
  sub_cell_map.push_back({9, 10, 12});
  sub_cell_map.push_back({10, 12, 13});
  sub_cell_map.push_back({10, 11, 13});
  sub_cell_map.push_back({12, 13, 14});
  #endif

  return sub_cell_map;
}
