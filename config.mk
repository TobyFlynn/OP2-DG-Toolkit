CONFIG_AR := ar rcs
CC = mpicc
CXX = mpicxx
NVCC = nvcc
HIPCC = hipcc

CUDA_ARCH = 70
HIP_ARCH = gfx908

OPENMP_FLAG := -fopenmp

OP2_DIR = /home/u1717021/Code/PhD/OP2-My-Fork/op2
OPENBLAS_DIR = /opt/OpenBLAS
ARMA_DIR = /home/u1717021/Code/PhD/armadillo-10.5.3/build
HDF5_DIR = /usr/local/module-software/hdf5-1.12.0-parallel
PETSC_DIR = /home/u1717021/Code/PhD/petsc-install
INIPP_DIR = /home/u1717021/Code/PhD/inipp/inipp
HIGHFIVE_DIR = /home/u1717021/Code/PhD/HighFive/include
MPI_DIR := /usr/local/module-software/openmpi-4.1.1
VTK_DIR := /usr/local
VTK_VERSION = 9.0
#HYPRE_DIR =

PART_LIB_NAME = PTSCOTCH
#PARMETIS_DIR = /dcs/pg20/u1717021/PhD/apps
PTSCOTCH_DIR = /home/u1717021/Code/PhD/scotch-install
#PARTITION_LIB = -L$(PARMETIS_DIR)/lib -lparmetis -lmetis -lGKlib
PARTITION_LIB = -L${PTSCOTCH_DIR}/lib -lptscotch -lscotch -lptscotcherr -lscotcherr -lptscotcherrexit -lscotcherrexit
#PARTITION_LIB = -L$(PARMETIS_DIR)/lib -lparmetis -lmetis -lGKlib -L${PTSCOTCH_DIR}/lib -lptscotch -lscotch -lptscotcherr -lscotcherr -lptscotcherrexit -lscotcherrexit

ORDER = 3
SOA = 1
BUILD_WITH_HYPRE = 0
BUILD_TOOLS = 1

# Probably do not need to change derived variables below this comment unless
# dependencies were installed in unusual locations

ifeq ($(BUILD_TOOLS),1)
	VTK_INC = -I$(VTK_DIR)/include/vtk-$(VTK_VERSION)
	VTK_LIB_PREFIX = lib
	VTK_LIB = -L$(VTK_DIR)/$(VTK_LIB_PREFIX) $(shell ls $(VTK_DIR)/$(VTK_LIB_PREFIX)/libvtk*-$(VTK_VERSION).so | sed "s+$(VTK_DIR)/$(VTK_LIB_PREFIX)/lib+-l+g" | sed "s+\.so++g")
endif

OP2_INC = -I$(OP2_DIR)/include
OPENBLAS_INC = -I$(OPENBLAS_DIR)/include
ARMA_INC = -I$(ARMA_DIR)/include
PETSC_INC = -I$(PETSC_DIR)/include
INIPP_INC = -I$(INIPP_DIR)
MPI_INC = -I$(MPI_DIR)/include
HIGHFIVE_INC = -I$(HIGHFIVE_DIR)
HYPRE_INC = -I$(HYPRE_DIR)/include

HDF5_LIB = -L$(HDF5_DIR)/lib -lhdf5
OP2_MPI_LIB = -L$(OP2_DIR)/lib -lop2_mpi

OP2_TRANSLATOR = $(OP2_DIR)/../translator/c/op2.py
