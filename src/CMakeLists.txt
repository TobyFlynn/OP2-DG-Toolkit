set(TOOLKIT_SRC utils/misc.cpp utils/nodes.cpp utils/polynomial.cpp utils/vandermonde.cpp utils/matrices.cpp utils/interpolate.cpp)
set(COMMON_SRC dg_op2_blas_op.cpp)
set(CPU_SRC "")
set(GPU_SRC "")

if(DIM EQUAL 2)
list(APPEND COMMON_SRC dg_constants/dg_constants_2d.cpp dg_mesh_op/dg_mesh_2d_op.cpp dg_operators_op/dg_operators_2d_op.cpp)
list(APPEND CPU_SRC dg_mesh/dg_mesh_2d_cpu.cpp)
list(APPEND GPU_SRC dg_mesh/dg_mesh_2d_gpu.cu)
elseif(DIM EQUAL 3)
list(APPEND COMMON_SRC dg_constants/dg_constants_3d.cpp dg_mesh_op/dg_mesh_3d_op.cpp dg_operators_op/dg_operators_3d_op.cpp)
endif()

# Only needs to be built once
if(DIM EQUAL 2)
  add_library(dgtoolkit SHARED ${TOOLKIT_SRC})
  target_link_libraries(dgtoolkit ${ARMA_LIB} ${OPENBLAS_LIB} ${EXTRA_LIBS})
  set(DG_LIBS dgtoolkit)
endif()

if(DIM EQUAL 2)
  set(DG_LIB_PREFIX op2dgtoolkit_2d)
elseif(DIM EQUAL 3)
  set(DG_LIB_PREFIX op2dgtoolkit_3d)
endif()

if(BUILD_SN AND BUILD_CPU)
  add_library(${DG_LIB_PREFIX}_seq SHARED ${COMMON_SRC} ${CPU_SRC} seq/dg_tookit_seqkernels.cpp)
  target_link_libraries(${DG_LIB_PREFIX}_seq dgtoolkit -L${OP2_DIR}/lib -lop2_seq -lop2_hdf5 ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} ${EXTRA_LIBS})
  target_compile_definitions(${DG_LIB_PREFIX}_seq PRIVATE DG_ORDER=${ORDER} DG_DIM=${DIM})

  add_library(${DG_LIB_PREFIX}_openmp SHARED ${COMMON_SRC} ${CPU_SRC} openmp/dg_tookit_kernels.cpp)
  target_link_libraries(${DG_LIB_PREFIX}_openmp dgtoolkit -L${OP2_DIR}/lib -lop2_openmp -lop2_hdf5 ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} ${EXTRA_LIBS})
  target_compile_definitions(${DG_LIB_PREFIX}_openmp PRIVATE DG_ORDER=${ORDER} DG_DIM=${DIM})

  list(APPEND DG_LIBS ${DG_LIB_PREFIX}_seq ${DG_LIB_PREFIX}_openmp)
endif()

if(BUILD_SN AND BUILD_GPU)
  add_library(${DG_LIB_PREFIX}_cuda SHARED ${COMMON_SRC} ${GPU_SRC} cuda/dg_tookit_kernels.cu)
  target_link_libraries(${DG_LIB_PREFIX}_cuda dgtoolkit -L${OP2_DIR}/lib -lop2_cuda -lop2_hdf5 ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} ${EXTRA_LIBS})
  target_compile_definitions(${DG_LIB_PREFIX}_cuda PRIVATE OP2_DG_CUDA DG_ORDER=${ORDER} DG_DIM=${DIM})

  list(APPEND DG_LIBS ${DG_LIB_PREFIX}_cuda)
endif()

if(BUILD_MPI AND BUILD_CPU)
  add_library(${DG_LIB_PREFIX}_mpi SHARED ${COMMON_SRC} ${CPU_SRC} seq/dg_tookit_seqkernels.cpp)
  target_link_libraries(${DG_LIB_PREFIX}_mpi dgtoolkit -L${OP2_DIR}/lib -lop2_mpi -lop2_hdf5 ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} ${PART_LIB} ${EXTRA_LIBS})
  target_compile_definitions(${DG_LIB_PREFIX}_mpi PRIVATE DG_ORDER=${ORDER} DG_DIM=${DIM})

  add_library(${DG_LIB_PREFIX}_mpi_openmp SHARED ${COMMON_SRC} ${CPU_SRC} openmp/dg_tookit_kernels.cpp)
  target_link_libraries(${DG_LIB_PREFIX}_mpi_openmp dgtoolkit -L${OP2_DIR}/lib -lop2_mpi -lop2_openmp -lop2_hdf5 ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} ${PART_LIB} ${EXTRA_LIBS})
  target_compile_definitions(${DG_LIB_PREFIX}_mpi_openmp PRIVATE DG_ORDER=${ORDER} DG_DIM=${DIM})

  list(APPEND DG_LIBS ${DG_LIB_PREFIX}_mpi ${DG_LIB_PREFIX}_mpi_openmp)
endif()

if(BUILD_MPI AND BUILD_GPU)
  add_library(${DG_LIB_PREFIX}_mpi_cuda SHARED ${COMMON_SRC} ${GPU_SRC} cuda/dg_tookit_kernels.cu)
  target_link_libraries(${DG_LIB_PREFIX}_mpi_cuda dgtoolkit -L${OP2_DIR}/lib -lop2_mpi_cuda -lop2_hdf5 ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} ${PART_LIB} ${EXTRA_LIBS})
  target_compile_definitions(${DG_LIB_PREFIX}_mpi_cuda PRIVATE OP2_DG_CUDA DG_ORDER=${ORDER} DG_DIM=${DIM})

  list(APPEND DG_LIBS ${DG_LIB_PREFIX}_mpi_cuda)
endif()

install(TARGETS ${DG_LIBS} DESTINATION lib)
