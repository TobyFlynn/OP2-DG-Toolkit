#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellIterator.h>
#include <vtkIdList.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <getopt.h>
#include <map>

#include "op_seq.h"

using namespace std;

// Stuff for parsing command line arguments
extern char *optarg;
extern int  optind, opterr, optopt;
static struct option options[] = {
  {"file", required_argument, 0, 0},
  {"out", required_argument, 0, 0},
  {"bc", required_argument, 0, 0},
  {0,    0,                  0,  0}
};

// Structs for converting mesh
struct Point2D {
  double x;
  double y;
};

struct Cell {
  int points[3];
};

struct Edge {
  int points[2];
  int cells[2];
  int num[2];
};

int getBoundaryEdgeNum(const string &type, double x0, double y0, double x1, double y1) {
  if(type == "cylinder_p") {
    if(x0 == 0.0 && x1 == 0.0) {
      // Inflow
      return 0;
    } else if(x0 == x1 && x0 > 5.0) {
      // Outflow
      return 1;
    } else if(x0 > 0.1 && x1 > 0.1 && x0 < 1.0 && x1 < 1.0
              && y0 > 0.1 && y1 > 0.1 && y0 < 0.9 && y1 < 0.9) {
      // Cylinder Wall
      return 2;
    } else {
      return -1;
    }
  }  else {
    cerr << "***ERROR*** Unrecognised boundary type specified" << endl;
  }
  return -1;
}

struct cmpCoords {
    bool operator()(const pair<double,double>& a, const pair<double,double>& b) const {
        bool xCmp = abs(a.first - b.first) < 1e-8;
        bool yCmp = abs(a.second - b.second) < 1e-8;
        if(xCmp && yCmp) {
          return false;
        }
        return a < b;
    }
};

int main(int argc, char **argv) {
  op_init(argc, argv, 1);
  string filename = "grid.vtk";
  string outdir = "./";
  string bcType = "";
  int opt_index = 0;
  while(getopt_long_only(argc, argv, "", options, &opt_index) != -1) {
    if(strcmp((char*)options[opt_index].name,"file") == 0) filename = optarg;
    if(strcmp((char*)options[opt_index].name,"out") == 0) outdir = optarg;
    if(strcmp((char*)options[opt_index].name,"bc") == 0) bcType = optarg;
  }

  if(outdir.back() != '/') {
    outdir += "/";
  }

  // Read in VTK file
  vtkSmartPointer<vtkUnstructuredGrid> grid;
  auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
  reader->SetFileName (filename.c_str());
  reader->Update();
  grid = reader->GetOutput();

  int numNodes = grid->GetNumberOfPoints();

  map<int,unique_ptr<Cell>> cellMap;
  map<pair<int,int>,unique_ptr<Edge>> internalEdgeMap;
  vector<int> pointIds;
  vector<double> x;
  vector<double> y;

  int numVTKCells = 0;
  int numVTKNodes = 0;
  vtkSmartPointer<vtkCellIterator> cellIterator = grid->NewCellIterator();
  while(!cellIterator->IsDoneWithTraversal()) {
    if(cellIterator->GetCellType() == VTK_TRIANGLE) {
      vtkSmartPointer<vtkIdList> ids = cellIterator->GetPointIds();
      int newIds[3];
      for(int i = 0; i < 3; i++) {
        auto it = find(pointIds.begin(), pointIds.end(), ids->GetId(i));
        if(it != pointIds.end()) {
          newIds[i] = distance(pointIds.begin(), it);
        } else {
          pointIds.push_back(ids->GetId(i));
          newIds[i] = pointIds.size() - 1;
          double coords[3];
          grid->GetPoint(ids->GetId(i), coords);
          x.push_back(coords[0]);
          y.push_back(coords[1]);
        }
      }

      // Add cell to map
      unique_ptr<Cell> cell = make_unique<Cell>();
      cell->points[0] = newIds[0];
      cell->points[1] = newIds[2];
      cell->points[2] = newIds[1];
      cellMap.insert(pair<int,unique_ptr<Cell>>(cellIterator->GetCellId(), move(cell)));
    }
    // Go to next cell
    cellIterator->GoToNextCell();
  }

  vector<int> elements;
  for(auto const &elem : cellMap) {
    elements.push_back(elem.second->points[0]);
    elements.push_back(elem.second->points[1]);
    elements.push_back(elem.second->points[2]);
  }

  cout << "Number of points: " << x.size() << endl;
  cout << "VTK Number of points: " << grid->GetNumberOfPoints() << endl;
  cout << "Number of cell: " << elements.size() / 3 << endl;

  // Add edges to edge map if not already contained in mapping
  // If already added then update cell field
  // Try both combinations
  for(int i = 0; i < elements.size() / 3; i++) {
    int ind = i * 3;
    // Check that points are anticlockwise
    int p_1 = elements[ind];
    int p_2 = elements[ind + 1];
    int p_3 = elements[ind + 2];
    double val = (x[p_2] - x[p_1]) * (y[p_3] - y[p_2]) - (y[p_2] - y[p_1]) * (x[p_3] - x[p_2]);
    if(val < 0) {
      cout << "Switching points" << endl;
      cout << "  Old val: " << val << endl;
      elements[ind] = p_3;
      elements[ind + 2] = p_1;
      p_1 = elements[ind];
      p_2 = elements[ind + 1];
      p_3 = elements[ind + 2];
      val = (x[p_2] - x[p_1]) * (y[p_3] - y[p_2]) - (y[p_2] - y[p_1]) * (x[p_3] - x[p_2]);
      cout << "  New val: " << val << endl;
    }
    for(int j = 0; j < 3; j++) {
      int p1, p2;
      if(j == 0) {
        p1 = elements[ind];
        p2 = elements[ind + 1];
      } else if(j == 1) {
        p1 = elements[ind + 1];
        p2 = elements[ind + 2];
      } else {
        p1 = elements[ind + 2];
        p2 = elements[ind];
      }

      pair<int,int> key;
      if(p1 < p2) {
        key = make_pair(p1, p2);
      } else {
        key = make_pair(p2, p1);
      }

      if(internalEdgeMap.count(key) == 0) {
        unique_ptr<Edge> edge = make_unique<Edge>();
        edge->points[0] = key.first;
        edge->points[1] = key.second;
        edge->cells[0] = i;
        edge->cells[1] = -1;
        edge->num[0] = j;
        edge->num[1] = -1;
        internalEdgeMap.insert(make_pair(key, move(edge)));
      } else {
          if(internalEdgeMap.at(key)->cells[1] != -1) {
            cout << "ERROR in edge mapping: " << endl;
            cout << "  Old values: " << internalEdgeMap.at(key)->cells[0] << " " << internalEdgeMap.at(key)->cells[1] << endl;
            cout << "  New Value: " << i << endl;
            cout << "  Edges: " << internalEdgeMap.at(key)->points[0] << " " << internalEdgeMap.at(key)->points[1] << endl;
            cout << "  Key: " << key.first << " " << key.second << endl;
          }
          if(internalEdgeMap.at(key)->points[0] != key.first || internalEdgeMap.at(key)->points[1] != key.second) {
            cout << "ERROR in edge mapping: " << endl;
            cout << "  Prev Edge Nodes: " << internalEdgeMap.at(key)->points[0] << " " << internalEdgeMap.at(key)->points[1] << endl;
            cout << "  Current Nodes: " << p1 << " " << p2 << endl;
            cout << "  Key: " << key.first << " " << key.second << endl;
          }
          internalEdgeMap.at(key)->cells[1] = i;
          internalEdgeMap.at(key)->num[1] = j;
      }
    }
  }

  // Do periodic boundary
  map<pair<double,double>,pair<int,int>,cmpCoords> periodic_map;
  if(bcType == "cylinder_p") {
    cout << "Doing periodic boundary conditions" << endl;
    for(auto const &edge : internalEdgeMap) {
      if(edge.second->cells[1] == -1) {
        double x0 = x[edge.second->points[0]];
        double y0 = y[edge.second->points[0]];
        double x1 = x[edge.second->points[1]];
        double y1 = y[edge.second->points[1]];
        if(y0 != y1) continue;
        double x_k0 = x0 < x1 ? x0 : x1;
        double x_k1 = x0 > x1 ? x0 : x1;

        if(periodic_map.count({x_k0, x_k1}) == 0) {
          periodic_map.insert({{x_k0, x_k1}, edge.first});
        } else {
          // Update one of the edges
          edge.second->cells[1] = internalEdgeMap.at(periodic_map.at({x_k0, x_k1}))->cells[0];
          edge.second->num[1] = internalEdgeMap.at(periodic_map.at({x_k0, x_k1}))->num[0];
          // Remove the other edge
          internalEdgeMap.erase(periodic_map.at({x_k0, x_k1}));
          periodic_map.erase({x_k0, x_k1});
        }
      }
    }
    cout << "Number of periodic edges that do not match up: " << periodic_map.size() << endl;
  }

  vector<int> edge2node_vec, edge2cell_vec, edgeNum_vec;
  vector<int> bedge2node_vec, bedge2cell_vec, bedgeNum_vec, bedgeType_vec;
  for(auto const &edge : internalEdgeMap) {
    if(edge.second->cells[1] == -1) {
      bedge2node_vec.push_back(edge.second->points[0]);
      bedge2node_vec.push_back(edge.second->points[1]);
      bedge2cell_vec.push_back(edge.second->cells[0]);
      bedgeNum_vec.push_back(edge.second->num[0]);
      double x0 = x[edge.second->points[0]];
      double y0 = y[edge.second->points[0]];
      double x1 = x[edge.second->points[1]];
      double y1 = y[edge.second->points[1]];
      int bType = getBoundaryEdgeNum(bcType, x0, y0, x1, y1);
      bedgeType_vec.push_back(bType);
    } else {
      if(edge.second->points[0] == edge.second->points[1])
        cout << "***** ERROR: Edge with identical points *****" << endl;
      if(edge.second->cells[0] == edge.second->cells[1])
        cout << "***** ERROR: Edge with identical cells *****" << endl;
      edge2node_vec.push_back(edge.second->points[0]);
      edge2node_vec.push_back(edge.second->points[1]);
      edge2cell_vec.push_back(edge.second->cells[0]);
      edge2cell_vec.push_back(edge.second->cells[1]);
      edgeNum_vec.push_back(edge.second->num[0]);
      edgeNum_vec.push_back(edge.second->num[1]);
    }
  }

  op_set nodes  = op_decl_set(x.size(), "nodes");
  op_set cells  = op_decl_set(elements.size() / 3, "cells");
  op_set edges  = op_decl_set(edgeNum_vec.size() / 2, "edges");
  op_set bedges = op_decl_set(bedgeNum_vec.size(), "bedges");

  op_map cell2nodes  = op_decl_map(cells, nodes, 3, elements.data(), "cell2nodes");
  op_map edge2nodes  = op_decl_map(edges, nodes, 2, edge2node_vec.data(), "edge2nodes");
  op_map edge2cells  = op_decl_map(edges, cells, 2, edge2cell_vec.data(), "edge2cells");
  op_map bedge2nodes = op_decl_map(bedges, nodes, 2, bedge2node_vec.data(), "bedge2nodes");
  op_map bedge2cells = op_decl_map(bedges, cells, 1, bedge2cell_vec.data(), "bedge2cells");

  vector<double> coords_vec;
  for(int i = 0; i < x.size(); i++) {
    coords_vec.push_back(x[i]);
    coords_vec.push_back(y[i]);
  }
  op_dat node_coords = op_decl_dat(nodes, 2, "double", coords_vec.data(), "node_coords");
  op_dat bedge_type  = op_decl_dat(bedges, 1, "int", bedgeType_vec.data(), "bedge_type");
  op_dat edgeNum     = op_decl_dat(edges, 2, "int", edgeNum_vec.data(), "edgeNum");
  op_dat bedgeNum    = op_decl_dat(bedges, 1, "int", bedgeNum_vec.data(), "bedgeNum");

  op_partition("" STRINGIFY(OP2_PARTITIONER), "KWAY", cells, edge2cells, NULL);

  std::string meshfile = outdir + "mesh.h5";
  op_dump_to_hdf5(meshfile.c_str());

  op_exit();
}
