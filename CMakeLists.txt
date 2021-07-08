cmake_minimum_required(VERSION 3.18)
project(libop2dgtoolkit CXX)

# Default OP2 location for me
if(NOT OP2_DIR)
  set(OP2_DIR "/home/u1717021/Code/PhD/OP2-Common/op2/c/build")
endif()

# Load OP2 configuration (Set OP2_DIR to the directory containing OP2Config.cmake)
find_package(OP2 REQUIRED PATHS
  ${OP2-APPS_SOURCE_DIR}/../../op2/c/build
  ${PROJECT_SOURCE_DIR}/../../../op2/c/build
  ${CMAKE_INSTALL_PREFIX}/lib/op2)

# Import compiler flags for all build types
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OP2_CXX_FLAGS} -std=c++14")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OP2_C_FLAGS}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_75")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_75 -rdc=true")

include_directories(${OP2_INCLUDE_DIRS})
add_definitions(${OP2_USER_DEFINITIONS})

include(OP2Application)

# OpenBLAS library for me
if(NOT OPENBLAS_DIR)
  set(OPENBLAS_DIR "/opt/OpenBLAS")
endif()

set(PARMETIS_DIR "/home/u1717021/Code/PhD/parmetis-4.0.3/build")

if(NOT ARMA_DIR)
  set(ARMA_DIR "/home/u1717021/Code/PhD/armadillo-10.5.3/build")
endif()

enable_language(CUDA)
find_package(CUDAToolkit)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})

# Install headers
file(GLOB HEADERS include/*.h)
install(FILES ${HEADERS} DESTINATION include)

# Invoke subprojects
add_subdirectory(src)
