#!/bin/bash

set -e

rm -rf build
rm -rf gen

mkdir -p gen/kernels
mkdir -p gen/openBLAS
mkdir -p gen/cuBLAS
mkdir -p gen/utils

python3 preprocessor.py 2

cd gen

python3 $OP2_TRANSLATOR \
  dg_mesh.cpp dg_op2_blas.cpp \
  dg_operators.cpp openBLAS/* \
  cuBLAS/* kernels/

cd ..

mkdir build

cd build

cmake .. \
  -DOP2_DIR=/dcs/pg20/u1717021/PhD/OP2-Common/op2 \
  -DOPENBLAS_DIR=/dcs/pg20/u1717021/PhD/apps \
  -DPART_LIB_NAME=PARMETIS \
  -DPARMETIS_DIR=/dcs/pg20/u1717021/PhD/apps \
  -DARMA_DIR=/dcs/pg20/u1717021/PhD/apps \
  -DBUILD_CPU=ON \
  -DBUILD_SN=ON \
  -DCMAKE_INSTALL_PREFIX=$(pwd)

make -j

make install