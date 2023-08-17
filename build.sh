#!/bin/bash

set -e

rm -rf build
rm -rf gen_2d
rm -rf gen_3d

mkdir gen_2d
cd gen_2d

mkdir -p kernels
mkdir -p utils
mkdir -p dg_constants
mkdir -p dg_mesh
mkdir -p matrices/2d
mkdir -p op2_utils
mkdir -p matrices/3d/custom_kernels
mkdir -p linear_solvers/amgx_amg
mkdir -p linear_solvers/hypre_amg
mkdir -p linear_solvers/petsc_amg
mkdir -p linear_solvers/petsc_utils
mkdir -p linear_solvers/petsc_block_jacobi
mkdir -p linear_solvers/petsc_jacobi/custom_kernels
mkdir -p linear_solvers/pmultigrid/custom_kernels
mkdir -p linear_solvers/petsc_pmultigrid
mkdir -p linear_solvers/petsc_inv_mass
mkdir -p linear_solvers/initial_guess_extrapolation
mkdir -p dg_operators/custom_kernels
mkdir -p blas/kernels

cd ..
mkdir gen_3d
cd gen_3d

mkdir -p kernels
mkdir -p utils
mkdir -p dg_constants
mkdir -p dg_mesh
mkdir -p op2_utils
mkdir -p matrices/2d
mkdir -p matrices/3d/custom_kernels
mkdir -p linear_solvers/amgx_amg
mkdir -p linear_solvers/petsc_amg
mkdir -p linear_solvers/hypre_amg
mkdir -p linear_solvers/petsc_utils
mkdir -p linear_solvers/petsc_block_jacobi
mkdir -p linear_solvers/petsc_jacobi/custom_kernels
mkdir -p linear_solvers/pmultigrid/custom_kernels
mkdir -p linear_solvers/petsc_pmultigrid
mkdir -p linear_solvers/petsc_inv_mass
mkdir -p linear_solvers/initial_guess_extrapolation
mkdir -p dg_operators/custom_kernels
mkdir -p blas/kernels

cd ..

ORDER=3
SOA=1

python3 preprocessor.py 2 $ORDER

python3 preprocessor.py 3 $ORDER

if [ $SOA = 1 ]; then
  export OP_AUTO_SOA=1
fi

cd gen_2d

python3 $OP2_TRANSLATOR \
  dg_tookit.cpp dg_mesh/dg_mesh_2d.cpp \
  blas/dg_op2_blas.cpp dg_operators/dg_operators_2d.cpp \
  matrices/poisson_matrix.cpp \
  matrices/poisson_coarse_matrix.cpp \
  matrices/poisson_semi_matrix_free.cpp \
  matrices/poisson_matrix_free_diag.cpp \
  matrices/poisson_matrix_free.cpp \
  matrices/2d/poisson_matrix_over_int.cpp \
  matrices/2d/poisson_coarse_matrix.cpp \
  matrices/2d/poisson_coarse_matrix_over_int.cpp \
  matrices/2d/poisson_matrix_free_mult.cpp \
  matrices/2d/poisson_matrix_free_diag.cpp \
  matrices/2d/poisson_semi_matrix_free_over_int.cpp \
  matrices/2d/poisson_matrix_free_over_int.cpp \
  matrices/2d/mm_poisson_matrix_over_int.cpp \
  matrices/2d/mm_poisson_matrix_free.cpp \
  matrices/2d/mm_poisson_matrix_free_over_int.cpp \
  matrices/2d/factor_poisson_matrix_free_mult.cpp \
  matrices/2d/factor_poisson_matrix_free_diag.cpp \
  matrices/2d/factor_mm_poisson_matrix_free_diag.cpp \
  matrices/2d/factor_poisson_coarse_matrix.cpp \
  matrices/2d/factor_poisson_matrix_over_int.cpp \
  matrices/2d/factor_poisson_coarse_matrix_over_int.cpp \
  matrices/2d/factor_poisson_semi_matrix_free_over_int.cpp \
  matrices/2d/factor_poisson_matrix_free_over_int.cpp \
  matrices/2d/factor_mm_poisson_matrix_over_int.cpp \
  matrices/2d/factor_mm_poisson_matrix_free_over_int.cpp \
  matrices/2d/cub_poisson_matrix.cpp \
  matrices/2d/cub_factor_poisson_matrix.cpp \
  matrices/2d/cub_mm_poisson_matrix.cpp \
  matrices/2d/cub_factor_mm_poisson_matrix.cpp \
  linear_solvers/petsc_block_jacobi/petsc_block_jacobi.cpp \
  linear_solvers/pmultigrid/pmultigrid.cpp \
  linear_solvers/petsc_inv_mass/petsc_inv_mass.cpp \
  linear_solvers/petsc_jacobi/petsc_jacobi.cpp \
  linear_solvers/initial_guess_extrapolation/initial_guess_extrapolation.cpp \
  kernels/

# Add compiler definitions to every kernel
sed -i "19i #include \"dg_compiler_defs.h\"" cuda/dg_tookit_kernels.cu
sed -i "1i #include \"dg_compiler_defs.h\"" openmp/dg_tookit_kernels.cpp
sed -i "1i #include \"dg_compiler_defs.h\"" seq/dg_tookit_seqkernels.cpp
sed -i "20i #include \"dg_global_constants/dg_mat_constants_2d.h\"" cuda/dg_tookit_kernels.cu
sed -i "2i #include \"dg_global_constants/dg_mat_constants_2d.h\"" openmp/dg_tookit_kernels.cpp
sed -i "2i #include \"dg_global_constants/dg_mat_constants_2d.h\"" seq/dg_tookit_seqkernels.cpp
sed -i "2i #include \"cblas.h\"" openmp/dg_tookit_kernels.cpp
sed -i "2i #include \"cblas.h\"" seq/dg_tookit_seqkernels.cpp

cd ../gen_3d

python3 $OP2_TRANSLATOR \
  dg_tookit.cpp dg_mesh/dg_mesh_3d.cpp \
  blas/dg_op2_blas.cpp dg_operators/dg_operators_3d.cpp \
  matrices/poisson_matrix.cpp \
  matrices/poisson_coarse_matrix.cpp \
  matrices/poisson_semi_matrix_free.cpp \
  matrices/poisson_matrix_free_diag.cpp \
  matrices/3d/poisson_matrix.cpp \
  matrices/3d/poisson_coarse_matrix.cpp \
  matrices/3d/poisson_semi_matrix_free.cpp \
  matrices/3d/poisson_matrix_free_mult.cpp \
  matrices/3d/poisson_matrix_free_diag.cpp \
  matrices/3d/mm_poisson_matrix.cpp \
  matrices/3d/mm_poisson_matrix_free.cpp \
  matrices/3d/factor_poisson_matrix.cpp \
  matrices/3d/factor_poisson_coarse_matrix.cpp \
  matrices/3d/factor_poisson_semi_matrix_free.cpp \
  matrices/3d/factor_poisson_matrix_free_diag.cpp \
  matrices/3d/factor_poisson_matrix_free_mult.cpp \
  matrices/3d/factor_mm_poisson_matrix.cpp \
  matrices/3d/factor_mm_poisson_semi_matrix_free.cpp \
  matrices/3d/factor_mm_poisson_matrix_free.cpp \
  matrices/3d/factor_mm_poisson_matrix_free_diag.cpp \
  linear_solvers/petsc_block_jacobi/petsc_block_jacobi.cpp \
  linear_solvers/pmultigrid/pmultigrid.cpp \
  linear_solvers/petsc_inv_mass/petsc_inv_mass.cpp \
  linear_solvers/petsc_jacobi/petsc_jacobi.cpp \
  linear_solvers/initial_guess_extrapolation/initial_guess_extrapolation.cpp \
  kernels/

# Add compiler definitions to every kernel
sed -i "16i #include \"dg_compiler_defs.h\"" cuda/dg_tookit_kernels.cu
sed -i "1i #include \"dg_compiler_defs.h\"" openmp/dg_tookit_kernels.cpp
sed -i "1i #include \"dg_compiler_defs.h\"" seq/dg_tookit_seqkernels.cpp
sed -i "17i #include \"dg_global_constants/dg_mat_constants_3d.h\"" cuda/dg_tookit_kernels.cu
sed -i "2i #include \"dg_global_constants/dg_mat_constants_3d.h\"" openmp/dg_tookit_kernels.cpp
sed -i "2i #include \"dg_global_constants/dg_mat_constants_3d.h\"" seq/dg_tookit_seqkernels.cpp
sed -i "2i #include \"cblas.h\"" openmp/dg_tookit_kernels.cpp
sed -i "2i #include \"cblas.h\"" seq/dg_tookit_seqkernels.cpp

cd ..

mkdir build

cd build

cmake .. \
  -DOP2_DIR=$OP2_DIR \
  -DOPENBLAS_DIR=$OPENBLAS_DIR \
  -DPART_LIB_NAME=PTSCOTCH \
  -DPTSCOTCH_DIR=$PTSCOTCH_DIR \
  -DARMA_DIR=$ARMA_DIR \
  -DHDF5_DIR=$HDF5_DIR \
  -DPETSC_DIR=$PETSC_DIR \
  -DINIPP_DIR=$INIPP_DIR \ # /path/to/repo/inipp/inipp
  -DHYPRE_DIR=/work/e01/e01/tflynne01/code/hypre-sp-install \
  -DAMGX_DIR=/work/e01/e01/tflynne01/code/AMGX-install \
  -DBUILD_CPU=OFF \
  -DBUILD_SN=ON \
  -DBUILD_MPI=ON \
  -DBUILD_GPU=ON \
  -DBUILD_TESTS=ON \
  -DORDER=$ORDER \
  -DSOA=$SOA \
  -DCMAKE_INSTALL_PREFIX=$(pwd)

make -j 4

make install
