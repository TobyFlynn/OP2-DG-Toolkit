#!/bin/bash

set -e

rm -rf build

./code_gen.sh

mkdir build

cd build

cmake .. \
  -DOP2_DIR=/dcs/pg20/u1717021/PhD/OP2-My-Fork/op2 \
  -DOPENBLAS_DIR=/dcs/pg20/u1717021/PhD/apps \
  -DPART_LIB_NAME=PARMETIS \
  -DPARMETIS_DIR=/dcs/pg20/u1717021/PhD/apps \
  -DARMA_DIR=/dcs/pg20/u1717021/PhD/apps \
  -DHDF5_DIR=/dcs/pg20/u1717021/PhD/apps \
  -DPETSC_DIR=/dcs/pg20/u1717021/PhD/petsc-install \
  -DINIPP_DIR=/dcs/pg20/u1717021/PhD/inipp/inipp \
  -DHIGHFIVE_DIR=/dcs/pg20/u1717021/PhD/HighFive/include \
  -DBUILD_CPU=ON \
  -DBUILD_SN=ON \
  -DBUILD_MPI=ON \
  -DBUILD_GPU=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_TOOLS=ON \
  -DORDER=$ORDER \
  -DSOA=$SOA \
  -DCMAKE_INSTALL_PREFIX=$(pwd)

make -j 4

make install
