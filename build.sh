#!/bin/bash

set -e

rm -rf build
rm -rf gen_2d
rm -rf gen_3d

mkdir -p gen_2d/kernels
mkdir -p gen_2d/utils
mkdir -p gen_2d/dg_constants
mkdir -p gen_2d/dg_mesh
mkdir -p gen_2d/dg_operators/custom_kernels
mkdir -p gen_2d/blas

mkdir -p gen_3d/kernels
mkdir -p gen_3d/utils
mkdir -p gen_3d/dg_constants
mkdir -p gen_3d/dg_mesh
mkdir -p gen_3d/dg_operators/custom_kernels
mkdir -p gen_3d/blas

ORDER=3

python3 preprocessor.py 2 $ORDER

python3 preprocessor.py 3 $ORDER

cd gen_2d

python3 $OP2_TRANSLATOR \
  dg_tookit.cpp dg_mesh/dg_mesh_2d.cpp \
  blas/dg_op2_blas.cpp dg_operators/dg_operators_2d.cpp \
  kernels/

# Add compiler definitions to every kernel
sed -i "19i #include \"dg_compiler_defs.h\"" cuda/dg_tookit_kernels.cu
sed -i "1i #include \"dg_compiler_defs.h\"" openmp/dg_tookit_kernels.cpp
sed -i "1i #include \"dg_compiler_defs.h\"" seq/dg_tookit_seqkernels.cpp
sed -i "2i #include \"cblas.h\"" openmp/dg_tookit_kernels.cpp
sed -i "2i #include \"cblas.h\"" seq/dg_tookit_seqkernels.cpp

cd ../gen_3d

OP_AUTO_SOA=1 python3 $OP2_TRANSLATOR \
  dg_tookit.cpp dg_mesh/dg_mesh_3d.cpp \
  blas/dg_op2_blas.cpp dg_operators/dg_operators_3d.cpp \
  kernels/

# Add compiler definitions to every kernel
sed -i "16i #include \"dg_compiler_defs.h\"" cuda/dg_tookit_kernels.cu
sed -i "1i #include \"dg_compiler_defs.h\"" openmp/dg_tookit_kernels.cpp
sed -i "1i #include \"dg_compiler_defs.h\"" seq/dg_tookit_seqkernels.cpp
sed -i "2i #include \"cblas.h\"" openmp/dg_tookit_kernels.cpp
sed -i "2i #include \"cblas.h\"" seq/dg_tookit_seqkernels.cpp

cd ..

mkdir build

cd build

cmake .. \
  -DOP2_DIR=/dcs/pg20/u1717021/PhD/OP2-Common/op2 \
  -DOPENBLAS_DIR=/dcs/pg20/u1717021/PhD/apps \
  -DPART_LIB_NAME=PARMETIS \
  -DPARMETIS_DIR=/dcs/pg20/u1717021/PhD/apps \
  -DARMA_DIR=/dcs/pg20/u1717021/PhD/apps \
  -DHDF5_DIR=/dcs/pg20/u1717021/PhD/apps \
  -DHIGHFIVE_DIR=/dcs/pg20/u1717021/PhD/HighFive/include \
  -DBUILD_CPU=ON \
  -DBUILD_SN=ON \
  -DBUILD_MPI=ON \
  -DBUILD_GPU=OFF \
  -DBUILD_TESTS=ON \
  -DORDER=$ORDER \
  -DCMAKE_INSTALL_PREFIX=$(pwd)

make -j

make install
