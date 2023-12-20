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
  dg_toolkit.cpp dg_mesh/dg_mesh_2d.cpp \
  blas/dg_op2_blas.cpp dg_operators/dg_operators_2d.cpp \
  matrices/poisson_matrix.cpp \
  matrices/poisson_coarse_matrix.cpp \
  matrices/poisson_matrix_free_diag.cpp \
  matrices/poisson_matrix_free_block_diag.cpp \
  matrices/2d/poisson_coarse_matrix.cpp \
  matrices/2d/poisson_matrix_free_mult.cpp \
  matrices/2d/poisson_matrix_free_diag.cpp \
  matrices/2d/poisson_matrix_free_block_diag.cpp \
  matrices/2d/mm_poisson_matrix_free.cpp \
  matrices/2d/factor_poisson_matrix_free_mult.cpp \
  matrices/2d/factor_poisson_matrix_free_diag.cpp \
  matrices/2d/factor_poisson_matrix_free_block_diag.cpp \
  matrices/2d/factor_poisson_matrix_free_mult_oi.cpp \
  matrices/2d/factor_poisson_matrix_free_diag_oi.cpp \
  matrices/2d/factor_mm_poisson_matrix_free_diag.cpp \
  matrices/2d/factor_mm_poisson_matrix_free_block_diag.cpp \
  matrices/2d/factor_poisson_coarse_matrix.cpp \
  linear_solvers/petsc_block_jacobi/petsc_block_jacobi.cpp \
  linear_solvers/pmultigrid/pmultigrid.cpp \
  linear_solvers/petsc_inv_mass/petsc_inv_mass.cpp \
  linear_solvers/petsc_jacobi/petsc_jacobi.cpp \
  linear_solvers/initial_guess_extrapolation/initial_guess_extrapolation.cpp \
  kernels/

# Add compiler definitions to every kernel
cuda_line_no_2d=$(grep -n op_cuda_reduction cuda/dg_toolkit_kernels.cu | cut -d : -f 1)
hip_line_no_2d=$(grep -n op_hip_reduction hip/dg_toolkit_kernels.cpp | cut -d : -f 1)
openmp_line_no_2d=$(grep -n op_lib_cpp openmp/dg_toolkit_kernels.cpp | cut -d : -f 1)
seq_line_no_2d=$(grep -n op_lib_cpp seq/dg_toolkit_seqkernels.cpp | cut -d : -f 1)

cuda_line_no_2d=$((cuda_line_no_2d+1))
hip_line_no_2d=$((hip_line_no_2d+1))
openmp_line_no_2d=$((openmp_line_no_2d+1))
seq_line_no_2d=$((seq_line_no_2d+1))

text_gpu_2d="#include \"dg_compiler_defs.h\"\n#include \"dg_global_constants/dg_mat_constants_2d.h\""
text_cpu_2d="#include \"dg_compiler_defs.h\"\n#include \"dg_global_constants/dg_mat_constants_2d.h\"\n#include \"cblas.h\""

sed -i "${cuda_line_no_2d}i $text_gpu_2d" cuda/dg_toolkit_kernels.cu
sed -i "${hip_line_no_2d}i $text_gpu_2d" hip/dg_toolkit_kernels.cpp
sed -i "${openmp_line_no_2d}i $text_cpu_2d" openmp/dg_toolkit_kernels.cpp
sed -i "${seq_line_no_2d}i $text_cpu_2d" seq/dg_toolkit_seqkernels.cpp

#sed -i "1i #include \"omp.h\"" openmp/dg_tookit_kernels.cpp

cd ../gen_3d

python3 $OP2_TRANSLATOR \
  dg_toolkit.cpp dg_mesh/dg_mesh_3d.cpp \
  blas/dg_op2_blas.cpp dg_operators/dg_operators_3d.cpp \
  matrices/poisson_matrix.cpp \
  matrices/poisson_coarse_matrix.cpp \
  matrices/poisson_matrix_free_diag.cpp \
  matrices/poisson_matrix_free_block_diag.cpp \
  matrices/3d/poisson_matrix.cpp \
  matrices/3d/poisson_coarse_matrix.cpp \
  matrices/3d/poisson_matrix_free_block_diag.cpp \
  matrices/3d/poisson_matrix_free_mult.cpp \
  matrices/3d/poisson_matrix_free_diag.cpp \
  matrices/3d/mm_poisson_matrix.cpp \
  matrices/3d/mm_poisson_matrix_free.cpp \
  matrices/3d/factor_poisson_matrix.cpp \
  matrices/3d/factor_poisson_coarse_matrix.cpp \
  matrices/3d/factor_poisson_matrix_free_mult.cpp \
  matrices/3d/factor_poisson_matrix_free_diag.cpp \
  matrices/3d/factor_poisson_matrix_free_block_diag.cpp \
  matrices/3d/factor_mm_poisson_matrix.cpp \
  matrices/3d/factor_mm_poisson_matrix_free.cpp \
  matrices/3d/factor_mm_poisson_matrix_free_diag.cpp \
  matrices/3d/factor_mm_poisson_matrix_free_block_diag.cpp \
  linear_solvers/petsc_block_jacobi/petsc_block_jacobi.cpp \
  linear_solvers/pmultigrid/pmultigrid.cpp \
  linear_solvers/petsc_inv_mass/petsc_inv_mass.cpp \
  linear_solvers/petsc_jacobi/petsc_jacobi.cpp \
  linear_solvers/initial_guess_extrapolation/initial_guess_extrapolation.cpp \
  kernels/

# Add compiler definitions to every kernel
cuda_line_no_3d=$(grep -n op_cuda_reduction cuda/dg_toolkit_kernels.cu | cut -d : -f 1)
hip_line_no_3d=$(grep -n op_hip_reduction hip/dg_toolkit_kernels.cpp | cut -d : -f 1)
openmp_line_no_3d=$(grep -n op_lib_cpp openmp/dg_toolkit_kernels.cpp | cut -d : -f 1)
seq_line_no_3d=$(grep -n op_lib_cpp seq/dg_toolkit_seqkernels.cpp | cut -d : -f 1)

cuda_line_no_3d=$((cuda_line_no_3d+1))
hip_line_no_3d=$((hip_line_no_3d+1))
openmp_line_no_3d=$((openmp_line_no_3d+1))
seq_line_no_3d=$((seq_line_no_3d+1))

text_gpu_3d="#include \"dg_compiler_defs.h\"\n#include \"dg_global_constants/dg_mat_constants_3d.h\""
text_cpu_3d="#include \"dg_compiler_defs.h\"\n#include \"dg_global_constants/dg_mat_constants_3d.h\"\n#include \"cblas.h\""

sed -i "${cuda_line_no_3d}i $text_gpu_3d" cuda/dg_toolkit_kernels.cu
sed -i "${hip_line_no_3d}i $text_gpu_3d" hip/dg_toolkit_kernels.cpp
sed -i "${openmp_line_no_3d}i $text_cpu_3d" openmp/dg_toolkit_kernels.cpp
sed -i "${seq_line_no_3d}i $text_cpu_3d" seq/dg_toolkit_seqkernels.cpp

#sed -i "1i #include \"omp.h\"" openmp/dg_tookit_kernels.cpp

cd ../..
