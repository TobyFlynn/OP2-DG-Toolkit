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

// Stuff for parsing command line arguments
extern char *optarg;
extern int  optind, opterr, optopt;
static struct option options[] = {
  {"file", required_argument, 0, 0},
  {"out", required_argument, 0, 0},
  {0,    0,                  0,  0}
};

int main(int argc, char **argv) {
  op_init(argc, argv, 1);
  std::string filename = "grid.vtk";
  std::string outdir = "./";
  int opt_index = 0;
  while(getopt_long_only(argc, argv, "", options, &opt_index) != -1) {
    if(strcmp((char*)options[opt_index].name,"file") == 0) filename = optarg;
    if(strcmp((char*)options[opt_index].name,"out") == 0) outdir = optarg;
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

  double *coords_data = (double *)malloc(3 * numNodes * sizeof(double));
  std::vector<int> cells_vec;
  std::vector<int> face2cell_vec;
  std::vector<int> face2node_vec;
  std::vector<int> bface2cell_vec;
  std::vector<int> bface2node_vec;
  std::map<int,int> vtkInd2op2Ind;
  std::vector<int> faceNum_vec;
  std::vector<int> bfaceNum_vec;
  std::vector<int> periodicFace_vec;

  std::map<std::vector<std::pair<double,double>>,int> x_periodic_bc_map, y_periodic_bc_map, z_periodic_bc_map;

  int currentFace = 0;
  int numVTKCells = 0;
  vtkSmartPointer<vtkCellIterator> cellIterator = grid->NewCellIterator();
  while(!cellIterator->IsDoneWithTraversal()) {
    if(cellIterator->GetCellType() == VTK_TETRA) {
      vtkInd2op2Ind.insert({cellIterator->GetCellId(), numVTKCells});
      vtkSmartPointer<vtkIdList> ids = cellIterator->GetPointIds();
      std::vector<int> bPtsIds;
      for(int i = 0 ; i < 4; i++) {
        int ind = ids->GetId(i);
        bPtsIds.push_back(ind);
        double coords[3];
        grid->GetPoint(ind, coords);
        coords_data[ind * 3] = coords[0];
        coords_data[ind * 3 + 1] = coords[1];
        coords_data[ind * 3 + 2] = coords[2];

        cells_vec.push_back(ind);
      }

      for(int i = 0; i < 4; i++) {
        vtkNew<vtkIdList> idList;
        for(int j = 0; j < 4; j++) {
          if(j != i)
            idList->InsertNextId(ids->GetId(j));
        }

        vtkSmartPointer<vtkIdList> neighbour = vtkSmartPointer<vtkIdList>::New();
        grid->GetCellNeighbors(cellIterator->GetCellId(), idList, neighbour);

        if(neighbour->GetNumberOfIds() == 1 && grid->GetCell(neighbour->GetId(0))->GetCellType() == VTK_TETRA) {
          // Make sure not to duplicate faces
          if(neighbour->GetId(0) < cellIterator->GetCellId()) {
            face2cell_vec.push_back(cellIterator->GetCellId());
            face2cell_vec.push_back(neighbour->GetId(0));

            if(i == 0)
              faceNum_vec.push_back(2);
            else if(i == 1)
              faceNum_vec.push_back(3);
            else if(i == 2)
              faceNum_vec.push_back(1);
            else
              faceNum_vec.push_back(0);

            // Get neighbour edge number
            vtkSmartPointer<vtkIdList> nIds = grid->GetCell(neighbour->GetId(0))->GetPointIds();
            if(idList->IsId(nIds->GetId(0)) == -1)
              faceNum_vec.push_back(2);
            else if(idList->IsId(nIds->GetId(1)) == -1)
              faceNum_vec.push_back(3);
            else if(idList->IsId(nIds->GetId(2)) == -1)
              faceNum_vec.push_back(1);
            else
              faceNum_vec.push_back(0);

            for(int pt = 0; pt < 3; pt++) {
              face2node_vec.push_back(idList->GetId(pt));
            }
            periodicFace_vec.push_back(0);
            currentFace++;
          }
        } else {
          double coords0[3], coords1[3], coords2[3];
          grid->GetPoint(idList->GetId(0), coords0);
          grid->GetPoint(idList->GetId(1), coords1);
          grid->GetPoint(idList->GetId(2), coords2);
          if(fabs(coords0[0] - coords1[0]) < 1e-8 && fabs(coords0[0] - coords2[0]) < 1e-8) {
            std::vector<std::pair<double,double>> key;
            key.push_back({coords0[1], coords0[2]});
            key.push_back({coords1[1], coords1[2]});
            key.push_back({coords2[1], coords2[2]});
            std::sort(key.begin(), key.end());
            auto it = x_periodic_bc_map.begin();
            while(it != x_periodic_bc_map.end()) {
              bool same = true;
              for(int i = 0; i < 3; i++) {
                same = same && fabs(it->first[i].first - key[i].first) < 1e-8;
                same = same && fabs(it->first[i].second - key[i].second) < 1e-8;
              }
              if(same)
                break;
              it++;
            }
            /*
            auto range = x_periodic_bc_map.equal_range(key);
            auto it = x_periodic_bc_map.end();
            if(range.first != x_periodic_bc_map.end()) {
              bool same = true;
              for(int i = 0; i < 3; i++) {
                same = same && fabs(range.first->first[i].first - key[i].first) < 1e-8;
                same = same && fabs(range.first->first[i].second - key[i].second) < 1e-8;
              }
              if(same) {
                it = range.first;
              }
            }
            if(range.second != x_periodic_bc_map.end()) {
              bool same = true;
              for(int i = 0; i < 3; i++) {
                same = same && fabs(range.second->first[i].first - key[i].first) < 1e-8;
                same = same && fabs(range.second->first[i].second - key[i].second) < 1e-8;
              }
              if(same) {
                it = range.second;
              }
            }
            */
            // auto it = x_periodic_bc_map.find(key);
            if(it == x_periodic_bc_map.end()) {
              // Corresponding BC face not found
              x_periodic_bc_map.insert({key, currentFace});
              face2cell_vec.push_back(cellIterator->GetCellId());
              face2cell_vec.push_back(-1);
              if(i == 0)
                faceNum_vec.push_back(2);
              else if(i == 1)
                faceNum_vec.push_back(3);
              else if(i == 2)
                faceNum_vec.push_back(1);
              else
                faceNum_vec.push_back(0);
              faceNum_vec.push_back(-1);
              for(int pt = 0; pt < 3; pt++) {
                face2node_vec.push_back(idList->GetId(pt));
              }
              periodicFace_vec.push_back(1);
              currentFace++;
            } else {
              // Corresponding BC face found
              face2cell_vec[it->second * 2 + 1] = cellIterator->GetCellId();
              if(i == 0)
                faceNum_vec[it->second * 2 + 1] = 2;
              else if(i == 1)
                faceNum_vec[it->second * 2 + 1] = 3;
              else if(i == 2)
                faceNum_vec[it->second * 2 + 1] = 1;
              else
                faceNum_vec[it->second * 2 + 1] = 0;
              x_periodic_bc_map.erase(it);
            }
          } else if(fabs(coords0[1] - coords1[1]) < 1e-8 && fabs(coords0[1] - coords2[1]) < 1e-8) {
            std::vector<std::pair<double,double>> key;
            key.push_back({coords0[0], coords0[2]});
            key.push_back({coords1[0], coords1[2]});
            key.push_back({coords2[0], coords2[2]});
            std::sort(key.begin(), key.end());
            auto it = y_periodic_bc_map.begin();
            while(it != y_periodic_bc_map.end()) {
              bool same = true;
              for(int i = 0; i < 3; i++) {
                same = same && fabs(it->first[i].first - key[i].first) < 1e-8;
                same = same && fabs(it->first[i].second - key[i].second) < 1e-8;
              }
              if(same)
                break;
              it++;
            }
            /*
            auto range = y_periodic_bc_map.equal_range(key);
            auto it = y_periodic_bc_map.end();
            if(range.first != y_periodic_bc_map.end()) {
              bool same = true;
              for(int i = 0; i < 3; i++) {
                same = same && fabs(range.first->first[i].first - key[i].first) < 1e-8;
                same = same && fabs(range.first->first[i].second - key[i].second) < 1e-8;
              }
              if(same) {
                it = range.first;
              }
            }
            if(range.second != y_periodic_bc_map.end()) {
              bool same = true;
              for(int i = 0; i < 3; i++) {
                same = same && fabs(range.second->first[i].first - key[i].first) < 1e-8;
                same = same && fabs(range.second->first[i].second - key[i].second) < 1e-8;
              }
              if(same) {
                it = range.second;
              }
            }
            */
            // auto it = y_periodic_bc_map.find(key);
            if(it == y_periodic_bc_map.end()) {
              // Corresponding BC face not found
              y_periodic_bc_map.insert({key, currentFace});
              face2cell_vec.push_back(cellIterator->GetCellId());
              face2cell_vec.push_back(-1);
              if(i == 0)
                faceNum_vec.push_back(2);
              else if(i == 1)
                faceNum_vec.push_back(3);
              else if(i == 2)
                faceNum_vec.push_back(1);
              else
                faceNum_vec.push_back(0);
              faceNum_vec.push_back(-1);
              for(int pt = 0; pt < 3; pt++) {
                face2node_vec.push_back(idList->GetId(pt));
              }
              periodicFace_vec.push_back(2);
              currentFace++;
            } else {
              // Corresponding BC face found
              face2cell_vec[it->second * 2 + 1] = cellIterator->GetCellId();
              if(i == 0)
                faceNum_vec[it->second * 2 + 1] = 2;
              else if(i == 1)
                faceNum_vec[it->second * 2 + 1] = 3;
              else if(i == 2)
                faceNum_vec[it->second * 2 + 1] = 1;
              else
                faceNum_vec[it->second * 2 + 1] = 0;
              y_periodic_bc_map.erase(it);
            }
          } else if(fabs(coords0[2] - coords1[2]) < 1e-8 && fabs(coords0[2] - coords2[2]) < 1e-8) {
            std::vector<std::pair<double,double>> key;
            key.push_back({coords0[0], coords0[1]});
            key.push_back({coords1[0], coords1[1]});
            key.push_back({coords2[0], coords2[1]});
            std::sort(key.begin(), key.end());
            auto it = z_periodic_bc_map.begin();
            while(it != z_periodic_bc_map.end()) {
              bool same = true;
              for(int i = 0; i < 3; i++) {
                same = same && fabs(it->first[i].first - key[i].first) < 1e-8;
                same = same && fabs(it->first[i].second - key[i].second) < 1e-8;
              }
              if(same)
                break;
              it++;
            }
            /*
            auto range = z_periodic_bc_map.equal_range(key);
            auto it = z_periodic_bc_map.end();
            if(range.first != z_periodic_bc_map.end()) {
              bool same = true;
              for(int i = 0; i < 3; i++) {
                same = same && fabs(range.first->first[i].first - key[i].first) < 1e-8;
                same = same && fabs(range.first->first[i].second - key[i].second) < 1e-8;
              }
              if(same) {
                it = range.first;
              }
            }
            if(range.second != z_periodic_bc_map.end()) {
              bool same = true;
              for(int i = 0; i < 3; i++) {
                same = same && fabs(range.second->first[i].first - key[i].first) < 1e-8;
                same = same && fabs(range.second->first[i].second - key[i].second) < 1e-8;
              }
              if(same) {
                it = range.second;
              }
            }
            */
            // auto it = z_periodic_bc_map.find(key);
            if(it == z_periodic_bc_map.end()) {
              // Corresponding BC face not found
              z_periodic_bc_map.insert({key, currentFace});
              face2cell_vec.push_back(cellIterator->GetCellId());
              face2cell_vec.push_back(-1);
              if(i == 0)
                faceNum_vec.push_back(2);
              else if(i == 1)
                faceNum_vec.push_back(3);
              else if(i == 2)
                faceNum_vec.push_back(1);
              else
                faceNum_vec.push_back(0);
              faceNum_vec.push_back(-1);
              for(int pt = 0; pt < 3; pt++) {
                face2node_vec.push_back(idList->GetId(pt));
              }
              periodicFace_vec.push_back(3);
              currentFace++;
            } else {
              // Corresponding BC face found
              face2cell_vec[it->second * 2 + 1] = cellIterator->GetCellId();
              if(i == 0)
                faceNum_vec[it->second * 2 + 1] = 2;
              else if(i == 1)
                faceNum_vec[it->second * 2 + 1] = 3;
              else if(i == 2)
                faceNum_vec[it->second * 2 + 1] = 1;
              else
                faceNum_vec[it->second * 2 + 1] = 0;
              z_periodic_bc_map.erase(it);
            }
          } else {
            // ERROR
            std::cout << "BC error" << std::endl;
          }

          // // Boundary edge
          // bface2cell_vec.push_back(cellIterator->GetCellId());
          // if(i == 0)
          //   bfaceNum_vec.push_back(2);
          // else if(i == 1)
          //   bfaceNum_vec.push_back(3);
          // else if(i == 2)
          //   bfaceNum_vec.push_back(1);
          // else
          //   bfaceNum_vec.push_back(0);
          //
          // for(int pt = 0; pt < 3; pt++) {
          //   bface2node_vec.push_back(idList->GetId(pt));
          // }
        }
      }
      numVTKCells++;
    }
    cellIterator->GoToNextCell();
  }

  int numCells = numVTKCells;
  int numFaces = face2cell_vec.size() / 2;
  int numBoundaryFaces = bface2cell_vec.size();

  std::cout << "Number of cells: " << numCells << std::endl;
  std::cout << "Number of faces: " << numFaces << std::endl;
  std::cout << "Number of boundary faces: " << numBoundaryFaces << std::endl;

  int numNeg = 0;
  for(int i = 0; i < face2cell_vec.size(); i++) {
    if(face2cell_vec[i] < 0) {
      numNeg++;
      std::cout << periodicFace_vec[i / 2];
    } else
      face2cell_vec[i] = vtkInd2op2Ind.at(face2cell_vec[i]);
  }
  std::cout << std::endl;
  for(int i = 0; i < bface2cell_vec.size(); i++) {
    bface2cell_vec[i] = vtkInd2op2Ind.at(bface2cell_vec[i]);
  }
  std::cout << "Number of negative inds: " << numNeg << std::endl;
  std::cout << "Size of x map: " << x_periodic_bc_map.size() << std::endl;
  std::cout << "Size of y map: " << y_periodic_bc_map.size() << std::endl;
  std::cout << "Size of z map: " << z_periodic_bc_map.size() << std::endl;

  // Optimise mapping order and indices
  std::vector<std::pair<std::pair<int,int>,int>> list;
  for(int i = 0; i < face2cell_vec.size() / 2; i++) {
    list.push_back({{face2cell_vec[i * 2], face2cell_vec[i * 2 + 1]}, i});
  }
  std::sort(list.begin(), list.end());
  std::vector<int> faceNum_tmp, periodicFace_tmp, face2node_tmp;
  for(int i = 0; i < list.size(); i++) {
    face2cell_vec[i * 2] = list[i].first.first;
    face2cell_vec[i * 2 + 1] = list[i].first.second;
    faceNum_tmp.push_back(faceNum_vec[list[i].second * 2]);
    faceNum_tmp.push_back(faceNum_vec[list[i].second * 2 + 1]);
    periodicFace_tmp.push_back(periodicFace_vec[list[i].second]);
    face2node_tmp.push_back(face2node_vec[list[i].second * 3]);
    face2node_tmp.push_back(face2node_vec[list[i].second * 3 + 1]);
    face2node_tmp.push_back(face2node_vec[list[i].second * 3 + 2]);
  }
  faceNum_vec = faceNum_tmp;
  periodicFace_vec = periodicFace_vec;
  face2node_vec = face2node_tmp;

  // Create OP2 objects
  op_set nodes  = op_decl_set(numNodes, "nodes");
  op_set cells  = op_decl_set(numCells, "cells");
  op_set faces  = op_decl_set(numFaces, "faces");
  op_set bfaces = op_decl_set(numBoundaryFaces, "bfaces");

  // Maps
  op_map cell2nodes  = op_decl_map(cells, nodes, 4, cells_vec.data(), "cell2nodes");
  op_map face2nodes  = op_decl_map(faces, nodes, 3, face2node_vec.data(), "face2nodes");
  op_map face2cells  = op_decl_map(faces, cells, 2, face2cell_vec.data(), "face2cells");
  op_map bface2nodes = op_decl_map(bfaces, nodes, 3, bface2node_vec.data(), "bface2nodes");
  op_map bface2cells = op_decl_map(bfaces, cells, 1, bface2cell_vec.data(), "bface2cells");

  // Dats
  op_dat node_coords  = op_decl_dat(nodes, 3, "double", coords_data, "node_coords");
  op_dat faceNum      = op_decl_dat(faces, 2, "int", faceNum_vec.data(), "faceNum");
  op_dat bfaceNum     = op_decl_dat(bfaces, 1, "int", bfaceNum_vec.data(), "bfaceNum");
  op_dat periodicFace = op_decl_dat(faces, 1, "int", periodicFace_vec.data(), "periodicFace");

  op_partition("" STRINGIFY(OP2_PARTITIONER), "KWAY", cells, face2cells, NULL);

  std::string meshfile = outdir + "mesh.h5";
  op_dump_to_hdf5(meshfile.c_str());

  free(coords_data);
  op_exit();
}
