#!/bin/bash

set -e

rm -rf build 

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
  -DBUILD_MPI=ON \
  -DCMAKE_INSTALL_PREFIX=$(pwd)

make

make install