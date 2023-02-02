set(OPENBLAS_SRC "")
set(CUBLAS_SRC "")

set(TOOLKIT_SRC utils/misc.cpp utils/nodes.cpp utils/polynomial.cpp utils/vandermonde.cpp utils/matrices.cpp utils/interpolate.cpp)
set(COMMON_SRC "")
set(CPU_SRC "")
set(GPU_SRC "")

if(DIM EQUAL 2)
list(APPEND COMMON_SRC dg_constants/dg_constants_2d.cpp dg_mesh_op/dg_mesh_2d_op.cpp dg_operators_op.cpp dg_op2_blas_op.cpp)
list(APPEND CPU_SRC dg_mesh/dg_mesh_2d_cpu.cpp)
list(APPEND GPU_SRC dg_mesh/dg_mesh_2d_gpu.cu)
list(APPEND OPENBLAS_SRC openBLAS_op/inv_op.cpp)
list(APPEND CUBLAS_SRC cuBLAS_op/inv_op.cu)
elseif(DIM EQUAL 3)
list(APPEND COMMON_SRC dg_constants/dg_constants_3d.cpp dg_mesh_op/dg_mesh_3d_op.cpp)
endif()

add_library(dgtoolkit SHARED ${TOOLKIT_SRC})
target_link_libraries(dgtoolkit ${ARMA_LIB} ${OPENBLAS_LIB} ${EXTRA_LIBS})
set(DG_LIBS dgtoolkit)

if(BUILD_SN AND BUILD_CPU)
  add_library(op2dgtoolkit_seq SHARED ${COMMON_SRC} ${CPU_SRC} ${OPENBLAS_SRC} seq/dg_tookit_seqkernels.cpp)
  target_link_libraries(op2dgtoolkit_seq dgtoolkit -L${OP2_DIR}/lib -lop2_seq -lop2_hdf5 ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} ${EXTRA_LIBS})
  target_compile_definitions(op2dgtoolkit_seq PRIVATE DG_ORDER=${ORDER} DG_DIM=${DIM})

  add_library(op2dgtoolkit_openmp SHARED ${COMMON_SRC} ${CPU_SRC} ${OPENBLAS_SRC} openmp/dg_tookit_kernels.cpp)
  target_link_libraries(op2dgtoolkit_openmp dgtoolkit -L${OP2_DIR}/lib -lop2_openmp -lop2_hdf5 ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} ${EXTRA_LIBS})
  target_compile_definitions(op2dgtoolkit_openmp PRIVATE DG_ORDER=${ORDER} DG_DIM=${DIM})

  list(APPEND DG_LIBS op2dgtoolkit_seq op2dgtoolkit_openmp)
endif()

if(BUILD_SN AND BUILD_GPU)
  add_library(op2dgtoolkit_cuda SHARED ${COMMON_SRC} ${GPU_SRC} ${CUBLAS_SRC} cuda/dg_tookit_kernels.cu)
  target_link_libraries(op2dgtoolkit_cuda dgtoolkit -L${OP2_DIR}/lib -lop2_cuda -lop2_hdf5 ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} CUDA::cudart_static CUDA::cublas ${EXTRA_LIBS})
  target_compile_definitions(op2dgtoolkit_cuda PRIVATE OP2_DG_CUDA DG_ORDER=${ORDER} DG_DIM=${DIM})

  list(APPEND DG_LIBS op2dgtoolkit_cuda)
endif()

if(BUILD_MPI AND BUILD_CPU)
  add_library(op2dgtoolkit_mpi SHARED ${COMMON_SRC} ${CPU_SRC} ${OPENBLAS_SRC} seq/dg_tookit_seqkernels.cpp)
  target_link_libraries(op2dgtoolkit_mpi dgtoolkit -L${OP2_DIR}/lib -lop2_mpi -lop2_hdf5 ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} ${PART_LIB} ${EXTRA_LIBS})
  target_compile_definitions(op2dgtoolkit_mpi PRIVATE DG_ORDER=${ORDER} DG_DIM=${DIM})

  add_library(op2dgtoolkit_mpi_openmp SHARED ${COMMON_SRC} ${CPU_SRC} ${OPENBLAS_SRC} openmp/dg_tookit_kernels.cpp)
  target_link_libraries(op2dgtoolkit_mpi_openmp dgtoolkit -L${OP2_DIR}/lib -lop2_mpi -lop2_openmp -lop2_hdf5 ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} ${PART_LIB} ${EXTRA_LIBS})
  target_compile_definitions(op2dgtoolkit_mpi_openmp PRIVATE DG_ORDER=${ORDER} DG_DIM=${DIM})

  list(APPEND DG_LIBS op2dgtoolkit_mpi op2dgtoolkit_mpi_openmp)
endif()

if(BUILD_MPI AND BUILD_GPU)
  add_library(op2dgtoolkit_mpi_cuda SHARED ${COMMON_SRC} ${GPU_SRC} ${CUBLAS_SRC} cuda/dg_tookit_kernels.cu)
  target_link_libraries(op2dgtoolkit_mpi_cuda dgtoolkit -L${OP2_DIR}/lib -lop2_mpi_cuda -lop2_hdf5 ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} CUDA::cudart_static CUDA::cublas ${PART_LIB} ${EXTRA_LIBS})
  target_compile_definitions(op2dgtoolkit_mpi_cuda PRIVATE OP2_DG_CUDA DG_ORDER=${ORDER} DG_DIM=${DIM})

  list(APPEND DG_LIBS op2dgtoolkit_mpi_cuda)
endif()

install(TARGETS ${DG_LIBS} DESTINATION lib)
