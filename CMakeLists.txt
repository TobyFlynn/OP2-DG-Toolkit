cmake_minimum_required(VERSION 3.18)
project(libop2dgtoolkit CXX)

# Import compiler flags for all build types
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_75")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_75 -rdc=true")

# Default OP2 location for me
if(NOT OP2_DIR)
  set(OP2_DIR "/home/u1717021/Code/PhD/OP2-Common/op2")
endif()

include_directories(${OP2_DIR}/include)

# OpenBLAS library for me
if(NOT OPENBLAS_DIR)
  set(OPENBLAS_DIR "/opt/OpenBLAS")
endif()

include_directories(${OPENBLAS_DIR}/include)

set(PARMETIS_DIR "/home/u1717021/Code/PhD/parmetis-4.0.3/build")

include_directories(${PARMETIS_DIR}/include)

if(NOT ARMA_DIR)
  set(ARMA_DIR "/home/u1717021/Code/PhD/armadillo-10.5.3/build")
endif()

include_directories(${ARMA_DIR}/include)

enable_language(CUDA)
find_package(CUDAToolkit)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})

# Install headers
file(GLOB HEADERS include/*.h)
install(FILES ${HEADERS} DESTINATION include)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Invoke subprojects
add_subdirectory(src)
add_subdirectory(tests)
