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
  std::map<int,int> vtkNodeInd2op2NodeInd;

  int numVTKCells = 0;
  int numVTKNodes = 0;
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
        if(vtkNodeInd2op2NodeInd.count(ind) == 0) {
          coords_data[numVTKNodes * 3] = coords[0];
          coords_data[numVTKNodes * 3 + 1] = coords[1];
          coords_data[numVTKNodes * 3 + 2] = coords[2];
          vtkNodeInd2op2NodeInd.insert({ind, numVTKNodes});
          numVTKNodes++;
        }

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
          }
        } else {
          // Boundary edge
          bface2cell_vec.push_back(cellIterator->GetCellId());
          if(i == 0)
            bfaceNum_vec.push_back(2);
          else if(i == 1)
            bfaceNum_vec.push_back(3);
          else if(i == 2)
            bfaceNum_vec.push_back(1);
          else
            bfaceNum_vec.push_back(0);

          for(int pt = 0; pt < 3; pt++) {
            bface2node_vec.push_back(idList->GetId(pt));
          }
        }
      }
      numVTKCells++;
    }
    cellIterator->GoToNextCell();
  }

  for(int i = 0; i < face2cell_vec.size(); i++) {
    face2cell_vec[i] = vtkInd2op2Ind.at(face2cell_vec[i]);
  }
  for(int i = 0; i < bface2cell_vec.size(); i++) {
    bface2cell_vec[i] = vtkInd2op2Ind.at(bface2cell_vec[i]);
  }

  for(int i = 0; i < cells_vec.size(); i++) {
    cells_vec[i] = vtkNodeInd2op2NodeInd.at(cells_vec[i]);
  }

  for(int i = 0; i < face2node_vec.size(); i++) {
    face2node_vec[i] = vtkNodeInd2op2NodeInd.at(face2node_vec[i]);
  }

  for(int i = 0; i < bface2node_vec.size(); i++) {
    bface2node_vec[i] = vtkNodeInd2op2NodeInd.at(bface2node_vec[i]);
  }

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
  int numCells = numVTKCells;
  int numFaces = face2cell_vec.size() / 2;
  int numBoundaryFaces = bface2cell_vec.size();

  std::cout << "Number of cells: " << numCells << std::endl;
  std::cout << "Number of faces: " << numFaces << std::endl;
  std::cout << "Number of boundary faces: " << numBoundaryFaces << std::endl;

  op_set nodes  = op_decl_set(numVTKNodes, "nodes");
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
