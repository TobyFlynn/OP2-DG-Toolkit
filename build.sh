#!/bin/bash

set -e

rm -rf build
rm -rf gen

mkdir -p gen/kernels
mkdir -p gen/openBLAS
mkdir -p gen/cuBLAS
mkdir -p gen/utils
mkdir -p gen/dg_constants
mkdir -p gen/dg_mesh
mkdir -p gen/dg_operators

python3 preprocessor.py 3
# python3 preprocessor_3d.py 3

cd gen

python3 $OP2_TRANSLATOR \
  dg_tookit.cpp dg_mesh/dg_mesh_2d.cpp \
  dg_op2_blas.cpp \
  dg_operators/dg_operators_2d.cpp openBLAS/* \
  cuBLAS/* kernels/
# python3 $OP2_TRANSLATOR \
#   dg_tookit.cpp \
#   dg_mesh/dg_mesh_3d.cpp kernels/

cd ..

sed -i "19i #include \"dg_compiler_defs.h\"" gen/cuda/dg_tookit_kernels.cu
sed -i "20i #include \"dg_global_constants.h\"" gen/cuda/dg_tookit_kernels.cu
sed -i "21i void set_cuda_constants_OP2_DG_CUDA() {" gen/cuda/dg_tookit_kernels.cu
sed -i "22i cutilSafeCall(cudaMemcpyToSymbol(FMASK_cuda, FMASK, DG_ORDER * DG_NPF * 3 * sizeof(int)));" gen/cuda/dg_tookit_kernels.cu
sed -i "23i cutilSafeCall(cudaMemcpyToSymbol(cubW_g_cuda, cubW_g, DG_ORDER * DG_CUB_NP * sizeof(double)));" gen/cuda/dg_tookit_kernels.cu
sed -i "24i cutilSafeCall(cudaMemcpyToSymbol(gaussW_g_cuda, gaussW_g, DG_ORDER * DG_GF_NP * sizeof(double)));" gen/cuda/dg_tookit_kernels.cu
sed -i "25i cutilSafeCall(cudaMemcpyToSymbol(DG_CONSTANTS_cuda, DG_CONSTANTS, DG_ORDER * 5 * sizeof(int)));" gen/cuda/dg_tookit_kernels.cu
sed -i "26i }" gen/cuda/dg_tookit_kernels.cu

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
  -DBUILD_GPU=ON \
  -DORDER=3 \
  -DDIM=2 \
  -DCMAKE_INSTALL_PREFIX=$(pwd)

make -j

make install
