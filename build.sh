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

sed -i "19i #include \"dg_compiler_defs.h\"" gen/cuda/dg_mesh_kernels.cu
sed -i "20i #include \"dg_global_constants.h\"" gen/cuda/dg_mesh_kernels.cu
sed -i "21i void set_cuda_constants_OP2_DG_CUDA() {" gen/cuda/dg_mesh_kernels.cu
sed -i "22i cutilSafeCall(cudaMemcpyToSymbol(FMASK_cuda, FMASK, DG_ORDER * DG_NPF * 3 * sizeof(int)));" gen/cuda/dg_mesh_kernels.cu
sed -i "23i cutilSafeCall(cudaMemcpyToSymbol(cubW_g_cuda, cubW_g, DG_ORDER * DG_CUB_NP * sizeof(double)));" gen/cuda/dg_mesh_kernels.cu
sed -i "24i cutilSafeCall(cudaMemcpyToSymbol(gaussW_g_cuda, gaussW_g, DG_ORDER * DG_GF_NP * sizeof(double)));" gen/cuda/dg_mesh_kernels.cu
sed -i "25i cutilSafeCall(cudaMemcpyToSymbol(DG_CONSTANTS_cuda, DG_CONSTANTS, DG_ORDER * 5 * sizeof(int)));" gen/cuda/dg_mesh_kernels.cu
sed -i "26i }" gen/cuda/dg_mesh_kernels.cu

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
