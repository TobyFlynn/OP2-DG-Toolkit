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

DIM=2
ORDER=3

python3 preprocessor.py $DIM $ORDER

cd gen

if [ $DIM -eq 2 ]
then
  python3 $OP2_TRANSLATOR \
    dg_tookit.cpp dg_mesh/dg_mesh_2d.cpp \
    dg_op2_blas.cpp dg_operators/dg_operators_2d.cpp \
    kernels/
elif [ $DIM -eq 3 ]
then
  python3 $OP2_TRANSLATOR \
    dg_tookit.cpp dg_mesh/dg_mesh_3d.cpp \
    dg_op2_blas.cpp kernels/
fi
cd ..

sed -i "19i #include \"dg_compiler_defs.h\"" gen/cuda/dg_tookit_kernels.cu
sed -i "20i #include \"dg_global_constants.h\"" gen/cuda/dg_tookit_kernels.cu
sed -i "21i void set_cuda_constants_OP2_DG_CUDA() {" gen/cuda/dg_tookit_kernels.cu
sed -i "22i cutilSafeCall(cudaMemcpyToSymbol(FMASK_cuda, FMASK, DG_ORDER * DG_NPF * 3 * sizeof(int)));" gen/cuda/dg_tookit_kernels.cu
sed -i "23i cutilSafeCall(cudaMemcpyToSymbol(cubW_g_cuda, cubW_g, DG_ORDER * DG_CUB_NP * sizeof(double)));" gen/cuda/dg_tookit_kernels.cu
sed -i "24i cutilSafeCall(cudaMemcpyToSymbol(gaussW_g_cuda, gaussW_g, DG_ORDER * DG_GF_NP * sizeof(double)));" gen/cuda/dg_tookit_kernels.cu
sed -i "25i cutilSafeCall(cudaMemcpyToSymbol(DG_CONSTANTS_cuda, DG_CONSTANTS, DG_ORDER * 5 * sizeof(int)));" gen/cuda/dg_tookit_kernels.cu
sed -i "26i }" gen/cuda/dg_tookit_kernels.cu

# Add compiler definitions to every kernel
sed -i "1i #include \"dg_compiler_defs.h\"" gen/openmp/dg_tookit_kernels.cpp
sed -i "1i #include \"dg_compiler_defs.h\"" gen/seq/dg_tookit_seqkernels.cpp

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
  -DDIM=$DIM \
  -DCMAKE_INSTALL_PREFIX=$(pwd)

make -j

make install
