CONFIG_AR := ar rcs
CC = mpicc
CXX = mpicxx
NVCC = nvcc
HIPCC = hipcc

CUDA_ARCH = 70
HIP_ARCH = gfx908

OPENMP_FLAG := -fopenmp

OP2_DIR = /dcs/pg20/u1717021/PhD/OP2-My-Fork/op2
OPENBLAS_DIR = /dcs/pg20/u1717021/PhD/apps
ARMA_DIR = /dcs/pg20/u1717021/PhD/apps
HDF5_DIR = /dcs/pg20/u1717021/PhD/apps
PETSC_DIR = /dcs/pg20/u1717021/PhD/petsc-install
INIPP_DIR = /dcs/pg20/u1717021/PhD/inipp/inipp
HIGHFIVE_DIR = /dcs/pg20/u1717021/PhD/HighFive/include
MPI_DIR := /dcs/pg20/u1717021/PhD/apps
VTK_DIR := /dcs/pg20/u1717021/PhD/apps

PART_LIB_NAME = PARMETIS
PARMETIS_DIR = /dcs/pg20/u1717021/PhD/apps
#PTSCOTCH_DIR = /dcs/pg20/u1717021/PhD/apps
PARTITION_LIB = -L$(PARMETIS_DIR)/lib -lparmetis -lmetis -lGKlib
#PARTITION_LIB = -L${PTSCOTCH_DIR}/lib -lptscotch -lscotch -lptscotcherr -lscotcherr -lptscotcherrexit -lscotcherrexit
VTK_VERSION = 9.0
#HYPRE_DIR =
#PARTITION_LIB = -L$(PARMETIS_DIR)/lib -lparmetis -lmetis -lGKlib -L${PTSCOTCH_DIR}/lib -lptscotch -lscotch -lptscotcherr -lscotcherr -lptscotcherrexit -lscotcherrexit

ORDER = 3
SOA = 1
BUILD_WITH_HYPRE = 0

# Probably do not need to change derived variables below this comment unless
# dependencies were installed in unusual locations

VTK_INC = -I$(VTK_DIR)/include/vtk-$(VTK_VERSION)
VTK_LIB_PREFIX = lib
VTK_LIB = -L$(VTK_DIR)/$(VTK_LIB_PREFIX) $(shell ls $(VTK_DIR)/$(VTK_LIB_PREFIX)/libvtk*-$(VTK_VERSION).so | sed "s+$(VTK_DIR)/$(VTK_LIB_PREFIX)/lib+-l+g" | sed "s+\.so++g")

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
