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

mkdir -p gen_3d/kernels
mkdir -p gen_3d/utils
mkdir -p gen_3d/dg_constants
mkdir -p gen_3d/dg_mesh
mkdir -p gen_3d/dg_operators/custom_kernels

ORDER=3

python3 preprocessor.py 2 $ORDER

python3 preprocessor.py 3 $ORDER

cd gen_2d

python3 $OP2_TRANSLATOR \
  dg_tookit.cpp dg_mesh/dg_mesh_2d.cpp \
  dg_op2_blas.cpp dg_operators/dg_operators_2d.cpp \
  kernels/

# CUDA OP2 library work around
sed -i "19i #include \"dg_compiler_defs.h\"" cuda/dg_tookit_kernels.cu
sed -i "20i #include \"dg_global_constants/dg_global_constants_2d.h\"" cuda/dg_tookit_kernels.cu
sed -i "21i void set_cuda_constants_OP2_DG_CUDA() {" cuda/dg_tookit_kernels.cu
sed -i "22i cutilSafeCall(cudaMemcpyToSymbol(FMASK_cuda, FMASK, DG_ORDER * DG_NPF * DG_NUM_FACES * sizeof(int)));" cuda/dg_tookit_kernels.cu
sed -i "23i cutilSafeCall(cudaMemcpyToSymbol(cubW_g_cuda, cubW_g, DG_ORDER * DG_CUB_NP * sizeof(DG_FP)));" cuda/dg_tookit_kernels.cu
sed -i "24i cutilSafeCall(cudaMemcpyToSymbol(gaussW_g_cuda, gaussW_g, DG_ORDER * DG_GF_NP * sizeof(DG_FP)));" cuda/dg_tookit_kernels.cu
sed -i "25i cutilSafeCall(cudaMemcpyToSymbol(DG_CONSTANTS_cuda, DG_CONSTANTS, DG_ORDER * DG_NUM_CONSTANTS * sizeof(int)));" cuda/dg_tookit_kernels.cu
sed -i "26i }" cuda/dg_tookit_kernels.cu

# Add compiler definitions to every kernel
sed -i "1i #include \"dg_compiler_defs.h\"" openmp/dg_tookit_kernels.cpp
sed -i "1i #include \"dg_compiler_defs.h\"" seq/dg_tookit_seqkernels.cpp
sed -i "2i #include \"cblas.h\"" openmp/dg_tookit_kernels.cpp
sed -i "2i #include \"cblas.h\"" seq/dg_tookit_seqkernels.cpp

cd ../gen_3d

python3 $OP2_TRANSLATOR \
  dg_tookit.cpp dg_mesh/dg_mesh_3d.cpp \
  dg_op2_blas.cpp dg_operators/dg_operators_3d.cpp \
  kernels/

# CUDA OP2 library work around
sed -i "16i #include \"dg_compiler_defs.h\"" cuda/dg_tookit_kernels.cu
sed -i "17i #include \"dg_global_constants/dg_global_constants_3d.h\"" cuda/dg_tookit_kernels.cu
sed -i "18i void set_cuda_constants_OP2_DG_3D_CUDA() {" cuda/dg_tookit_kernels.cu
sed -i "19i cutilSafeCall(cudaMemcpyToSymbol(FMASK_cuda, FMASK, DG_ORDER * DG_NPF * DG_NUM_FACES * sizeof(int)));" cuda/dg_tookit_kernels.cu
sed -i "20i cutilSafeCall(cudaMemcpyToSymbol(DG_CONSTANTS_cuda, DG_CONSTANTS, DG_ORDER * DG_NUM_CONSTANTS * sizeof(int)));" cuda/dg_tookit_kernels.cu
sed -i "21i }" cuda/dg_tookit_kernels.cu

# Add compiler definitions to every kernel
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
