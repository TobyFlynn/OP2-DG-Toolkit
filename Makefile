MAKEFILES_DIR != dirname $(realpath \
  $(word $(words $(MAKEFILE_LIST)), $(MAKEFILE_LIST)))
ROOT_DIR != realpath $(MAKEFILES_DIR)

INC := -I$(ROOT_DIR)/include
LIB := $(ROOT_DIR)/lib
OBJ := $(ROOT_DIR)/obj
BIN := $(ROOT_DIR)/bin
CODE_GEN_DIR := $(ROOT_DIR)/code_gen
BASE_OBJ_DIR := $(OBJ)/base
2D_SEQ_OBJ_DIR := $(OBJ)/2d_seq
2D_OMP_OBJ_DIR := $(OBJ)/2d_omp
2D_CUDA_OBJ_DIR := $(OBJ)/2d_cuda
2D_HIP_OBJ_DIR := $(OBJ)/2d_hip
2D_MPI_SEQ_OBJ_DIR := $(OBJ)/2d_mpi_seq
2D_MPI_OMP_OBJ_DIR := $(OBJ)/2d_mpi_omp
2D_MPI_CUDA_OBJ_DIR := $(OBJ)/2d_mpi_cuda
2D_MPI_HIP_OBJ_DIR := $(OBJ)/2d_mpi_hip
3D_SEQ_OBJ_DIR := $(OBJ)/3d_seq
3D_OMP_OBJ_DIR := $(OBJ)/3d_omp
3D_CUDA_OBJ_DIR := $(OBJ)/3d_cuda
3D_HIP_OBJ_DIR := $(OBJ)/3d_hip
3D_MPI_SEQ_OBJ_DIR := $(OBJ)/3d_mpi_seq
3D_MPI_OMP_OBJ_DIR := $(OBJ)/3d_mpi_omp
3D_MPI_CUDA_OBJ_DIR := $(OBJ)/3d_mpi_cuda
3D_MPI_HIP_OBJ_DIR := $(OBJ)/3d_mpi_hip

include config.mk

# Common compile flags
BASE_FLAGS := -g -O3
SEQ_CPU_COMPILER_FLAGS := $(BASE_FLAGS)
OMP_CPU_COMPILER_FLAGS := $(BASE_FLAGS) $(OPENMP_FLAG)
CUDA_COMPILER_FLAGS := $(BASE_FLAGS) $(OPENMP_FLAG)
NVCC_FLAGS := $(BASE_FLAGS) -rdc=true -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -Xcompiler $(OPENMP_FLAG) $(MPI_INC)
HIP_FLAGS := $(BASE_FLAGS) -fgpu-rdc --offload-arch=$(HIP_ARCH) $(OPENMP_FLAG) $(MPI_INC)
COMMON_COMPILE_DEFS_2D := -DDG_ORDER=$(ORDER) -DDG_DIM=2 -DDG_COL_MAJ -DMAX_CONST_SIZE=1024
COMMON_COMPILE_DEFS_3D := -DDG_ORDER=$(ORDER) -DDG_DIM=3 -DDG_COL_MAJ -DMAX_CONST_SIZE=1024
HIP_COMPILE_DEFS := -DOP2_DG_HIP -DDG_OP2_SOA
CUDA_COMPILE_DEFS := -DOP2_DG_CUDA -DDG_OP2_SOA
MPI_COMPILER_DEFS := -DDG_MPI
OP2_DG_TOOLKIT_INC := $(ARMA_INC) $(INC) $(OP2_INC) $(OPENBLAS_INC) $(PETSC_INC) $(INIPP_INC)
ifeq ($(BUILD_WITH_HYPRE),1)
	OP2_DG_TOOLKIT_INC := $(OP2_DG_TOOLKIT_INC) $(HYPRE_INC)
	COMMON_COMPILE_DEFS_2D := $(COMMON_COMPILE_DEFS_2D) -DINS_BUILD_WITH_HYPRE
	COMMON_COMPILE_DEFS_3D := $(COMMON_COMPILE_DEFS_3D) -DINS_BUILD_WITH_HYPRE
endif
TOOLS_INC := $(INC) $(VTK_INC) $(HIGHFIVE_INC) $(OP2_INC)

all: base 2d 3d

base: $(LIB)/libdgtoolkit.a

2d: cpu_2d cuda_2d

3d: cpu_3d cuda_3d

cpu: cpu_2d cpu_3d

cpu_2d: base $(LIB)/libop2dgtoolkit_2d_seq.a $(LIB)/libop2dgtoolkit_2d_openmp.a \
	$(LIB)/libop2dgtoolkit_2d_mpi.a $(LIB)/libop2dgtoolkit_2d_mpi_openmp.a

cpu_3d: base $(LIB)/libop2dgtoolkit_3d_seq.a $(LIB)/libop2dgtoolkit_3d_openmp.a \
	$(LIB)/libop2dgtoolkit_3d_mpi.a $(LIB)/libop2dgtoolkit_3d_mpi_openmp.a

cuda: cuda_2d cuda_3d

cuda_2d: base $(LIB)/libop2dgtoolkit_2d_cuda.a $(LIB)/libop2dgtoolkit_2d_mpi_cuda.a

cuda_3d: base $(LIB)/libop2dgtoolkit_3d_cuda.a $(LIB)/libop2dgtoolkit_3d_mpi_cuda.a

hip: hip_2d hip_3d

hip_2d: base $(LIB)/libop2dgtoolkit_2d_hip.a $(LIB)/libop2dgtoolkit_2d_mpi_hip.a

hip_3d: base $(LIB)/libop2dgtoolkit_3d_hip.a $(LIB)/libop2dgtoolkit_3d_mpi_hip.a

codegen: $(CODE_GEN_DIR)

tools: $(BIN)/hdf52vtk_2D $(BIN)/hdf52vtk_3D $(BIN)/vtk2hdf5_2D \
	$(BIN)/vtk2hdf5_3D $(BIN)/vtk2hdf5_periodic_cube_3D

clean:
	-rm -rf $(OBJ)
	-rm -rf $(LIB)
	-rm -rf $(BIN)
	-rm -rf $(CODE_GEN_DIR)

$(CODE_GEN_DIR):
	OP2_TRANSLATOR=$(OP2_TRANSLATOR) ORDER=$(ORDER) SOA=$(SOA) ./code_gen.sh


$(BIN):
	@mkdir -p $@

# Object directories
$(OBJ):
	@mkdir -p $@

$(BASE_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find src -maxdepth 1 -mindepth 1 | grep -v "\." | sed "s+src/++g"),$@/$(dir))

$(2D_SEQ_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find $(CODE_GEN_DIR)/gen_2d -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+$(CODE_GEN_DIR)/gen_2d/++g"),$@/$(dir))

$(2D_OMP_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find $(CODE_GEN_DIR)/gen_2d -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+$(CODE_GEN_DIR)/gen_2d/++g"),$@/$(dir))

$(2D_CUDA_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find $(CODE_GEN_DIR)/gen_2d -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+$(CODE_GEN_DIR)/gen_2d/++g"),$@/$(dir))

$(2D_HIP_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find $(CODE_GEN_DIR)/gen_2d -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+$(CODE_GEN_DIR)/gen_2d/++g"),$@/$(dir))

$(2D_MPI_SEQ_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find $(CODE_GEN_DIR)/gen_2d -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+$(CODE_GEN_DIR)/gen_2d/++g"),$@/$(dir))

$(2D_MPI_OMP_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find $(CODE_GEN_DIR)/gen_2d -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+$(CODE_GEN_DIR)/gen_2d/++g"),$@/$(dir))

$(2D_MPI_CUDA_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find $(CODE_GEN_DIR)/gen_2d -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+$(CODE_GEN_DIR)/gen_2d/++g"),$@/$(dir))

$(2D_MPI_HIP_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find $(CODE_GEN_DIR)/gen_2d -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+$(CODE_GEN_DIR)/gen_2d/++g"),$@/$(dir))

$(3D_SEQ_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find $(CODE_GEN_DIR)/gen_3d -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+$(CODE_GEN_DIR)/gen_3d/++g"),$@/$(dir))

$(3D_OMP_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find $(CODE_GEN_DIR)/gen_3d -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+$(CODE_GEN_DIR)/gen_3d/++g"),$@/$(dir))

$(3D_CUDA_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find $(CODE_GEN_DIR)/gen_3d -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+$(CODE_GEN_DIR)/gen_3d/++g"),$@/$(dir))

$(3D_HIP_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find $(CODE_GEN_DIR)/gen_3d -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+$(CODE_GEN_DIR)/gen_3d/++g"),$@/$(dir))

$(3D_MPI_SEQ_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find $(CODE_GEN_DIR)/gen_3d -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+$(CODE_GEN_DIR)/gen_3d/++g"),$@/$(dir))

$(3D_MPI_OMP_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find $(CODE_GEN_DIR)/gen_3d -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+$(CODE_GEN_DIR)/gen_3d/++g"),$@/$(dir))

$(3D_MPI_CUDA_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find $(CODE_GEN_DIR)/gen_3d -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+$(CODE_GEN_DIR)/gen_3d/++g"),$@/$(dir))

$(3D_MPI_HIP_OBJ_DIR): $(OBJ)
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find $(CODE_GEN_DIR)/gen_3d -maxdepth 3 -mindepth 1 | grep -v "\." | sed "s+$(CODE_GEN_DIR)/gen_3d/++g"),$@/$(dir))

$(LIB):
	@mkdir -p $@

# Basic DG library objects
DG_TOOLKIT_OBJ := $(addprefix $(BASE_OBJ_DIR)/utils/,\
	interpolate.o \
	matrices.o \
	misc.o \
	nodes.o \
	polynomial.o \
	vandermonde.o)

# Common objects
COMMON_OBJ := blas_op/dg_op2_blas_op.o \
	dg_dat_pool.o \
	timing.o \
	matrices_op/poisson_matrix_op.o \
	matrices_op/poisson_coarse_matrix_op.o \
	matrices_op/poisson_semi_matrix_free_op.o \
	matrices/poisson_matrix_free.o \
	matrices_op/poisson_matrix_free_diag_op.o \
	config.o

# MPI objects
MPI_OBJ := mpi_helper_func.o

# CPU objects
CPU_OBJ := op2_utils/utils_cpu.o \
	matrices/poisson_matrix_cpu.o \
	matrices/poisson_coarse_matrix_cpu.o

# CUDA objects
CUDA_OBJ := op2_utils/utils_gpu.o \
	matrices/poisson_matrix_gpu.o \
	matrices/poisson_coarse_matrix_gpu.o

# HIP objects
HIP_OBJ := op2_utils/utils_hip.o \
	matrices/poisson_matrix_hip.o \
	matrices/poisson_coarse_matrix_hip.o

# SOA specific object files
ifeq ($(SOA),1)
	CUDA_OBJ += blas/dg_op2_gemv_kernel_soa_gpu.o
	HIP_OBJ += blas/dg_op2_gemv_kernel_soa_hip.o
else
	CUDA_OBJ += blas/dg_op2_gemv_kernel_gpu.o
	HIP_OBJ += blas/dg_op2_gemv_kernel_hip.o
endif

# Linear solver object files
LINEAR_SOLVER_OBJ := linear_solvers/linear_solver.o \
	linear_solvers/petsc_amg/petsc_amg.o \
	linear_solvers/petsc_amg/petsc_amg_coarse.o \
	linear_solvers_op/petsc_block_jacobi/petsc_block_jacobi_op.o \
	linear_solvers_op/pmultigrid/pmultigrid_op.o \
	linear_solvers/petsc_pmultigrid/petsc_pmultigrid.o \
	linear_solvers_op/petsc_inv_mass/petsc_inv_mass_op.o \
	linear_solvers_op/petsc_jacobi/petsc_jacobi_op.o \
	linear_solvers_op/initial_guess_extrapolation/initial_guess_extrapolation_op.o
ifeq ($(BUILD_WITH_HYPRE),1)
	LINEAR_SOLVER_OBJ += linear_solvers/hypre_amg/hypre_amg.o
endif

# Linear solver CPU object files
LINEAR_SOLVER_CPU_OBJ := linear_solvers/petsc_utils/utils_cpu.o

# Linear solver CUDA object files
LINEAR_SOLVER_CUDA_OBJ := linear_solvers/petsc_utils/utils_gpu.o
ifeq ($(AMGX),true)
	LINEAR_SOLVER_CUDA_OBJ += linear_solvers/amgx_amg/amgx_amg.o
endif

# Linear solver HIP object files
LINEAR_SOLVER_HIP_OBJ := linear_solvers/petsc_utils/utils_hip.o

# 2D Common object files
COMMON_2D_OBJ := dg_constants/dg_constants_2d.o \
	dg_mesh_op/dg_mesh_2d_op.o \
	dg_operators_op/dg_operators_2d_op.o \
	matrices_op/2d/poisson_coarse_matrix_op.o \
	matrices_op/2d/poisson_matrix_free_mult_op.o \
	matrices_op/2d/poisson_matrix_free_diag_op.o \
	matrices/2d/poisson_matrix_free.o \
	matrices_op/2d/mm_poisson_matrix_free_op.o \
	matrices_op/2d/factor_poisson_matrix_free_mult_op.o \
	matrices_op/2d/factor_poisson_matrix_free_diag_op.o \
	matrices_op/2d/factor_poisson_matrix_free_mult_oi_op.o \
	matrices_op/2d/factor_poisson_matrix_free_diag_oi_op.o \
	matrices_op/2d/factor_mm_poisson_matrix_free_diag_op.o \
	matrices_op/2d/factor_poisson_coarse_matrix_op.o \
	$(COMMON_OBJ) \
	$(LINEAR_SOLVER_OBJ)

# 2D CPU only object files
CPU_2D_OBJ := dg_constants/dg_constants_2d_cpu.o \
	$(CPU_OBJ) \
	$(LINEAR_SOLVER_CPU_OBJ)

# 2D CUDA only object files
CUDA_2D_OBJ := dg_constants/dg_constants_2d_gpu.o \
	$(CUDA_OBJ) \
	$(LINEAR_SOLVER_CUDA_OBJ)

# 2D HIP only object files
HIP_2D_OBJ := dg_constants/dg_constants_2d_hip.o \
	$(HIP_OBJ) \
	$(LINEAR_SOLVER_HIP_OBJ)

# 3D Common object files
COMMON_3D_OBJ := dg_constants/dg_constants_3d.o \
	dg_mesh_op/dg_mesh_3d_op.o \
	dg_operators_op/dg_operators_3d_op.o \
	matrices_op/3d/poisson_matrix_op.o \
	matrices_op/3d/poisson_coarse_matrix_op.o \
	matrices_op/3d/poisson_semi_matrix_free_op.o \
	matrices/3d/poisson_matrix_free.o \
	matrices_op/3d/poisson_matrix_free_mult_op.o \
	matrices_op/3d/poisson_matrix_free_diag_op.o \
	matrices_op/3d/mm_poisson_matrix_op.o \
	matrices_op/3d/mm_poisson_matrix_free_op.o \
	matrices_op/3d/factor_poisson_matrix_op.o \
	matrices_op/3d/factor_poisson_coarse_matrix_op.o \
	matrices_op/3d/factor_poisson_semi_matrix_free_op.o \
	matrices_op/3d/factor_poisson_matrix_free_diag_op.o \
	matrices_op/3d/factor_poisson_matrix_free_diag_oi_op.o \
	matrices/3d/factor_poisson_matrix_free.o \
	matrices_op/3d/factor_poisson_matrix_free_mult_op.o \
	matrices_op/3d/factor_poisson_matrix_free_mult_oi_op.o \
	matrices_op/3d/factor_mm_poisson_matrix_op.o \
	matrices_op/3d/factor_mm_poisson_semi_matrix_free_op.o \
	matrices_op/3d/factor_mm_poisson_matrix_free_op.o \
	matrices_op/3d/factor_mm_poisson_matrix_free_diag_op.o \
	$(COMMON_OBJ) \
	$(LINEAR_SOLVER_OBJ)

# 3D CPU only object files
CPU_3D_OBJ := dg_mesh/dg_mesh_3d_cpu.o \
	dg_constants/dg_constants_3d_cpu.o \
	$(CPU_OBJ) \
	$(LINEAR_SOLVER_CPU_OBJ)

# 3D CUDA only object files
CUDA_3D_OBJ := dg_constants/dg_constants_3d_gpu.o \
	dg_mesh/dg_mesh_3d_gpu.o \
	dg_operators/custom_kernels/custom_grad_3d.o \
	dg_operators/custom_kernels/custom_mass.o \
	$(CUDA_OBJ) \
	$(LINEAR_SOLVER_CUDA_OBJ)

# 3D HIP only object files
HIP_3D_OBJ := dg_constants/dg_constants_3d_hip.o \
	dg_mesh/dg_mesh_3d_hip.o \
	$(HIP_OBJ) \
	$(LINEAR_SOLVER_HIP_OBJ)

# Generic rules src dir
DG_TOOLKIT_INC := $(ARMA_INC) $(INC)
$(BASE_OBJ_DIR)/%.o: src/%.cpp | $(BASE_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(DG_TOOLKIT_INC) -c $< -o $@

# Basic DG library
$(LIB)/libdgtoolkit.a: $(DG_TOOLKIT_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# Generic rules 2D seq
$(2D_SEQ_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_2d/%.cpp | $(2D_SEQ_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(SEQ_CPU_COMPILER_FLAGS) $(COMMON_COMPILE_DEFS_2D) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

# Generic rules 2D openmp
$(2D_OMP_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_2d/%.cpp | $(2D_OMP_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OMP_CPU_COMPILER_FLAGS) $(COMMON_COMPILE_DEFS_2D) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

# Generic rules 2D CUDA
$(2D_CUDA_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_2d/%.cpp | $(2D_CUDA_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(CUDA_COMPILER_FLAGS) $(COMMON_COMPILE_DEFS_2D) $(CUDA_COMPILE_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

$(2D_CUDA_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_2d/%.cu | $(2D_CUDA_OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) $(COMMON_COMPILE_DEFS_2D) $(CUDA_COMPILE_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

# Generic rules 2D HIP
$(2D_HIP_OBJ_DIR)/hip/dg_tookit_kernels.o: $(CODE_GEN_DIR)/gen_2d/hip/dg_tookit_kernels.cpp | $(2D_HIP_OBJ_DIR)
	$(HIPCC) $(HIP_FLAGS) $(COMMON_COMPILE_DEFS_2D) $(HIP_COMPILE_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

$(2D_HIP_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_2d/%.cpp | $(2D_HIP_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(HIP_COMPILER_FLAGS) $(COMMON_COMPILE_DEFS_2D) $(HIP_COMPILE_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

$(2D_HIP_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_2d/%.hip | $(2D_HIP_OBJ_DIR)
	$(HIPCC) $(HIP_FLAGS) $(COMMON_COMPILE_DEFS_2D) $(HIP_COMPILE_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

# Generic rules 2D MPI seq
$(2D_MPI_SEQ_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_2d/%.cpp | $(2D_MPI_SEQ_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(SEQ_CPU_COMPILER_FLAGS) $(COMMON_COMPILE_DEFS_2D) $(MPI_COMPILER_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

# Generic rules 2D MPI openmp
$(2D_MPI_OMP_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_2d/%.cpp | $(2D_MPI_OMP_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OMP_CPU_COMPILER_FLAGS) $(COMMON_COMPILE_DEFS_2D) $(MPI_COMPILER_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

# Generic rules 2D MPI CUDA
$(2D_MPI_CUDA_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_2d/%.cpp | $(2D_MPI_CUDA_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(CUDA_COMPILER_FLAGS) $(COMMON_COMPILE_DEFS_2D) $(CUDA_COMPILE_DEFS) $(MPI_COMPILER_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

$(2D_MPI_CUDA_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_2d/%.cu | $(2D_MPI_CUDA_OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) $(COMMON_COMPILE_DEFS_2D) $(CUDA_COMPILE_DEFS) $(MPI_COMPILER_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

# Generic rules 2D MPI HIP
$(2D_MPI_HIP_OBJ_DIR)/hip/dg_tookit_kernels.o: $(CODE_GEN_DIR)/gen_2d/hip/dg_tookit_kernels.cpp | $(2D_MPI_HIP_OBJ_DIR)
	$(HIPCC) $(HIP_FLAGS) $(COMMON_COMPILE_DEFS_2D) $(HIP_COMPILE_DEFS) $(MPI_COMPILER_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

$(2D_MPI_HIP_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_2d/%.cpp | $(2D_MPI_HIP_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(HIP_COMPILER_FLAGS) $(COMMON_COMPILE_DEFS_2D) $(HIP_COMPILE_DEFS) $(MPI_COMPILER_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

$(2D_MPI_HIP_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_2d/%.hip | $(2D_MPI_HIP_OBJ_DIR)
	$(HIPCC) $(HIP_FLAGS) $(COMMON_COMPILE_DEFS_2D) $(HIP_COMPILE_DEFS) $(MPI_COMPILER_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

# Generic rules 3D seq
$(3D_SEQ_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_3d/%.cpp | $(3D_SEQ_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(SEQ_CPU_COMPILER_FLAGS) $(COMMON_COMPILE_DEFS_3D) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

# Generic rules 3D openmp
$(3D_OMP_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_3d/%.cpp | $(3D_OMP_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OMP_CPU_COMPILER_FLAGS) $(COMMON_COMPILE_DEFS_3D) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

# Generic rules 3D CUDA
$(3D_CUDA_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_3d/%.cpp | $(3D_CUDA_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(CUDA_COMPILER_FLAGS) $(COMMON_COMPILE_DEFS_3D) $(CUDA_COMPILE_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

$(3D_CUDA_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_3d/%.cu | $(3D_CUDA_OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) $(COMMON_COMPILE_DEFS_3D) $(CUDA_COMPILE_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

# Generic rules 3D HIP
$(3D_HIP_OBJ_DIR)/hip/dg_tookit_kernels.o: $(CODE_GEN_DIR)/gen_3d/hip/dg_tookit_kernels.cpp | $(3D_HIP_OBJ_DIR)
	$(HIPCC) $(HIP_FLAGS) $(COMMON_COMPILE_DEFS_3D) $(HIP_COMPILE_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

$(3D_HIP_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_3d/%.cpp | $(3D_HIP_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(HIP_COMPILER_FLAGS) $(COMMON_COMPILE_DEFS_3D) $(HIP_COMPILE_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

$(3D_HIP_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_3d/%.hip | $(3D_HIP_OBJ_DIR)
	$(HIPCC) $(HIP_FLAGS) $(COMMON_COMPILE_DEFS_3D) $(HIP_COMPILE_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

# Generic rules 3D MPI seq
$(3D_MPI_SEQ_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_3d/%.cpp | $(3D_MPI_SEQ_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(SEQ_CPU_COMPILER_FLAGS) $(COMMON_COMPILE_DEFS_3D) $(MPI_COMPILER_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

# Generic rules 3D MPI openmp
$(3D_MPI_OMP_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_3d/%.cpp | $(3D_MPI_OMP_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OMP_CPU_COMPILER_FLAGS) $(COMMON_COMPILE_DEFS_3D) $(MPI_COMPILER_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

# Generic rules 3D MPI CUDA
$(3D_MPI_CUDA_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_3d/%.cpp | $(3D_MPI_CUDA_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(CUDA_COMPILER_FLAGS) $(COMMON_COMPILE_DEFS_3D) $(CUDA_COMPILE_DEFS) $(MPI_COMPILER_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

$(3D_MPI_CUDA_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_3d/%.cu | $(3D_MPI_CUDA_OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) $(COMMON_COMPILE_DEFS_3D) $(CUDA_COMPILE_DEFS) $(MPI_COMPILER_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

# Generic rules 3D MPI HIP
$(3D_MPI_HIP_OBJ_DIR)/hip/dg_tookit_kernels.o: $(CODE_GEN_DIR)/gen_3d/hip/dg_tookit_kernels.cpp | $(3D_MPI_HIP_OBJ_DIR)
	$(HIPCC) $(HIP_FLAGS) $(COMMON_COMPILE_DEFS_3D) $(HIP_COMPILE_DEFS) $(MPI_COMPILER_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

$(3D_MPI_HIP_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_3d/%.cpp | $(3D_MPI_HIP_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(HIP_COMPILER_FLAGS) $(COMMON_COMPILE_DEFS_3D) $(HIP_COMPILE_DEFS) $(MPI_COMPILER_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

$(3D_MPI_HIP_OBJ_DIR)/%.o: $(CODE_GEN_DIR)/gen_3d/%.hip | $(3D_MPI_HIP_OBJ_DIR)
	$(HIPCC) $(HIP_FLAGS) $(COMMON_COMPILE_DEFS_3D) $(HIP_COMPILE_DEFS) $(MPI_COMPILER_DEFS) $(OP2_DG_TOOLKIT_INC) -c $< -o $@

# 2D Seq OP2-DG library
2D_SEQ_OBJ := $(addprefix $(2D_SEQ_OBJ_DIR)/,\
	$(COMMON_2D_OBJ) \
	$(CPU_2D_OBJ) \
	seq/dg_tookit_seqkernels.o)
$(LIB)/libop2dgtoolkit_2d_seq.a: $(2D_SEQ_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# 2D OpenMP OP2-DG library
2D_OMP_OBJ := $(addprefix $(2D_OMP_OBJ_DIR)/,\
	$(COMMON_2D_OBJ) \
	$(CPU_2D_OBJ) \
	openmp/dg_tookit_kernels.o)
$(LIB)/libop2dgtoolkit_2d_openmp.a: $(2D_OMP_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# 2D CUDA OP2-DG library
2D_CUDA_OBJ := $(addprefix $(2D_CUDA_OBJ_DIR)/,\
	$(COMMON_2D_OBJ) \
	$(CUDA_2D_OBJ) \
	cuda/dg_tookit_kernels.o)
$(LIB)/libop2dgtoolkit_2d_cuda.a: $(2D_CUDA_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# 2D HIP OP2-DG library
2D_HIP_OBJ := $(addprefix $(2D_HIP_OBJ_DIR)/,\
	$(COMMON_2D_OBJ) \
	$(HIP_2D_OBJ) \
	hip/dg_tookit_kernels.o)
$(LIB)/libop2dgtoolkit_2d_hip.a: $(2D_HIP_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# 2D MPI Seq OP2-DG library
2D_MPI_SEQ_OBJ := $(addprefix $(2D_MPI_SEQ_OBJ_DIR)/,\
	$(COMMON_2D_OBJ) \
	$(CPU_2D_OBJ) \
	$(MPI_OBJ) \
	seq/dg_tookit_seqkernels.o)
$(LIB)/libop2dgtoolkit_2d_mpi.a: $(2D_MPI_SEQ_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# 2D MPI OpenMP OP2-DG library
2D_MPI_OMP_OBJ := $(addprefix $(2D_MPI_OMP_OBJ_DIR)/,\
	$(COMMON_2D_OBJ) \
	$(CPU_2D_OBJ) \
	$(MPI_OBJ) \
	openmp/dg_tookit_kernels.o)
$(LIB)/libop2dgtoolkit_2d_mpi_openmp.a: $(2D_MPI_OMP_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# 2D MPI CUDA OP2-DG library
2D_MPI_CUDA_OBJ := $(addprefix $(2D_MPI_CUDA_OBJ_DIR)/,\
	$(COMMON_2D_OBJ) \
	$(CUDA_2D_OBJ) \
	$(MPI_OBJ) \
	cuda/dg_tookit_kernels.o)
$(LIB)/libop2dgtoolkit_2d_mpi_cuda.a: $(2D_MPI_CUDA_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# 2D MPI HIP OP2-DG library
2D_MPI_HIP_OBJ := $(addprefix $(2D_MPI_HIP_OBJ_DIR)/,\
	$(COMMON_2D_OBJ) \
	$(HIP_2D_OBJ) \
	$(MPI_OBJ) \
	hip/dg_tookit_kernels.o)
$(LIB)/libop2dgtoolkit_2d_mpi_hip.a: $(2D_MPI_HIP_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# 3D Seq OP2-DG library
3D_SEQ_OBJ := $(addprefix $(3D_SEQ_OBJ_DIR)/,\
	$(COMMON_3D_OBJ) \
	$(CPU_3D_OBJ) \
	seq/dg_tookit_seqkernels.o)
$(LIB)/libop2dgtoolkit_3d_seq.a: $(3D_SEQ_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# 3D OpenMP OP2-DG library
3D_OMP_OBJ := $(addprefix $(3D_OMP_OBJ_DIR)/,\
	$(COMMON_3D_OBJ) \
	$(CPU_3D_OBJ) \
	openmp/dg_tookit_kernels.o)
$(LIB)/libop2dgtoolkit_3d_openmp.a: $(3D_OMP_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# 3D CUDA OP2-DG library
3D_CUDA_OBJ := $(addprefix $(3D_CUDA_OBJ_DIR)/,\
	$(COMMON_3D_OBJ) \
	$(CUDA_3D_OBJ) \
	cuda/dg_tookit_kernels.o)
$(LIB)/libop2dgtoolkit_3d_cuda.a: $(3D_CUDA_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# 3D HIP OP2-DG library
3D_HIP_OBJ := $(addprefix $(3D_HIP_OBJ_DIR)/,\
	$(COMMON_3D_OBJ) \
	$(HIP_3D_OBJ) \
	hip/dg_tookit_kernels.o)
$(LIB)/libop2dgtoolkit_3d_hip.a: $(3D_HIP_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# 3D MPI Seq OP2-DG library
3D_MPI_SEQ_OBJ := $(addprefix $(3D_MPI_SEQ_OBJ_DIR)/,\
	$(COMMON_3D_OBJ) \
	$(CPU_3D_OBJ) \
	$(MPI_OBJ) \
	seq/dg_tookit_seqkernels.o)
$(LIB)/libop2dgtoolkit_3d_mpi.a: $(3D_MPI_SEQ_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# 3D OpenMP OP2-DG library
3D_MPI_OMP_OBJ := $(addprefix $(3D_MPI_OMP_OBJ_DIR)/,\
	$(COMMON_3D_OBJ) \
	$(CPU_3D_OBJ) \
	$(MPI_OBJ) \
	openmp/dg_tookit_kernels.o)
$(LIB)/libop2dgtoolkit_3d_mpi_openmp.a: $(3D_MPI_OMP_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# 3D MPI CUDA OP2-DG library
3D_MPI_CUDA_OBJ := $(addprefix $(3D_MPI_CUDA_OBJ_DIR)/,\
	$(COMMON_3D_OBJ) \
	$(CUDA_3D_OBJ) \
	$(MPI_OBJ) \
	cuda/dg_tookit_kernels.o)
$(LIB)/libop2dgtoolkit_3d_mpi_cuda.a: $(3D_MPI_CUDA_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# 3D MPI HIP OP2-DG library
3D_MPI_HIP_OBJ := $(addprefix $(3D_MPI_HIP_OBJ_DIR)/,\
	$(COMMON_3D_OBJ) \
	$(HIP_3D_OBJ) \
	$(MPI_OBJ) \
	hip/dg_tookit_kernels.o)
$(LIB)/libop2dgtoolkit_3d_mpi_hip.a: $(3D_MPI_HIP_OBJ) | $(LIB)
	$(CONFIG_AR) $@ $^

# Tools
HDF52VTK_LIBS := $(HDF5_LIB) $(VTK_LIB)
VTK2HDF5_LIBS := $(VTK_LIB) $(OP2_MPI_LIB) $(HDF5_LIB) $(PARTITION_LIB)
$(BIN)/hdf52vtk_2D: tools/hdf52vtk_2D.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(COMMON_COMPILE_DEFS_2D) $(TOOLS_INC) $< $(HDF52VTK_LIBS) -o $@
$(BIN)/hdf52vtk_3D: tools/hdf52vtk_3D.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(COMMON_COMPILE_DEFS_3D) $(TOOLS_INC) $< $(HDF52VTK_LIBS) -o $@
$(BIN)/vtk2hdf5_2D: tools/vtk2hdf5_2D.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(COMMON_COMPILE_DEFS_2D) -DOP2_PARTITIONER=$(PART_LIB_NAME) $(TOOLS_INC) $< $(VTK2HDF5_LIBS) -o $@
$(BIN)/vtk2hdf5_3D: tools/vtk2hdf5_3D.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(COMMON_COMPILE_DEFS_3D) -DOP2_PARTITIONER=$(PART_LIB_NAME) $(TOOLS_INC) $< $(VTK2HDF5_LIBS) -o $@
$(BIN)/vtk2hdf5_periodic_cube_3D: tools/vtk2hdf5_periodic_cube_3D.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(COMMON_COMPILE_DEFS_3D) -DOP2_PARTITIONER=$(PART_LIB_NAME) $(TOOLS_INC) $< $(VTK2HDF5_LIBS) -o $@
