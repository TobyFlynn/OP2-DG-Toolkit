#!/bin/bash

set -e

rm -rf gen_2d
rm -rf gen_3d

dir_list=$(find src -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+src/++g")

mkdir -p code_gen/gen_2d
cd code_gen/gen_2d

for i in $dir_list
do
  mkdir -p $i
done

cd ..
mkdir gen_3d
cd gen_3d

for i in $dir_list
do
  mkdir -p $i
done

cd ../..

python3 preprocessor.py 2 $ORDER

python3 preprocessor.py 3 $ORDER

if [ $SOA = 1 ]; then
  export OP_AUTO_SOA=1
fi

cd code_gen/gen_2d

python3 $OP2_TRANSLATOR \
  dg_tookit.cpp dg_mesh/dg_mesh_2d.cpp \
  blas/dg_op2_blas.cpp dg_operators/dg_operators_2d.cpp \
  matrices/poisson_matrix.cpp \
  matrices/poisson_coarse_matrix.cpp \
  matrices/poisson_semi_matrix_free.cpp \
  matrices/poisson_matrix_free_diag.cpp \
  matrices/poisson_matrix_free.cpp \
  matrices/2d/poisson_coarse_matrix.cpp \
  matrices/2d/poisson_matrix_free_mult.cpp \
  matrices/2d/poisson_matrix_free_diag.cpp \
  matrices/2d/mm_poisson_matrix_free.cpp \
  matrices/2d/factor_poisson_matrix_free_mult.cpp \
  matrices/2d/factor_poisson_matrix_free_diag.cpp \
  matrices/2d/factor_mm_poisson_matrix_free_diag.cpp \
  matrices/2d/factor_poisson_coarse_matrix.cpp \
  linear_solvers/petsc_block_jacobi/petsc_block_jacobi.cpp \
  linear_solvers/pmultigrid/pmultigrid.cpp \
  linear_solvers/petsc_inv_mass/petsc_inv_mass.cpp \
  linear_solvers/petsc_jacobi/petsc_jacobi.cpp \
  linear_solvers/initial_guess_extrapolation/initial_guess_extrapolation.cpp \
  kernels/

# Add compiler definitions to every kernel
sed -i "17i #include \"dg_compiler_defs.h\"" cuda/dg_tookit_kernels.cu
sed -i "17i #include \"dg_compiler_defs.h\"" hip/dg_tookit_kernels.cpp
sed -i "1i #include \"dg_compiler_defs.h\"" openmp/dg_tookit_kernels.cpp
sed -i "1i #include \"omp.h\"" openmp/dg_tookit_kernels.cpp
sed -i "1i #include \"dg_compiler_defs.h\"" seq/dg_tookit_seqkernels.cpp
sed -i "18i #include \"dg_global_constants/dg_mat_constants_2d.h\"" cuda/dg_tookit_kernels.cu
sed -i "18i #include \"dg_global_constants/dg_mat_constants_2d.h\"" hip/dg_tookit_kernels.cpp
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
sed -i "16i #include \"dg_compiler_defs.h\"" hip/dg_tookit_kernels.cpp
sed -i "1i #include \"dg_compiler_defs.h\"" openmp/dg_tookit_kernels.cpp
sed -i "1i #include \"omp.h\"" openmp/dg_tookit_kernels.cpp
sed -i "1i #include \"dg_compiler_defs.h\"" seq/dg_tookit_seqkernels.cpp
sed -i "17i #include \"dg_global_constants/dg_mat_constants_3d.h\"" cuda/dg_tookit_kernels.cu
sed -i "17i #include \"dg_global_constants/dg_mat_constants_3d.h\"" hip/dg_tookit_kernels.cpp
sed -i "2i #include \"dg_global_constants/dg_mat_constants_3d.h\"" openmp/dg_tookit_kernels.cpp
sed -i "2i #include \"dg_global_constants/dg_mat_constants_3d.h\"" seq/dg_tookit_seqkernels.cpp
sed -i "2i #include \"cblas.h\"" openmp/dg_tookit_kernels.cpp
sed -i "2i #include \"cblas.h\"" seq/dg_tookit_seqkernels.cpp

cd ../..
