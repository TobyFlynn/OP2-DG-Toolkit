set(TOOLKIT_SRC utils/misc.cpp utils/nodes.cpp utils/polynomial.cpp utils/vandermonde.cpp utils/matrices.cpp utils/interpolate.cpp)
set(COMMON_SRC blas_op/dg_op2_blas_op.cpp)
set(CPU_SRC "")
if(SOA EQUAL 1)
  set(GPU_SRC blas/dg_op2_gemv_kernel_soa.cu)
else()
  set(GPU_SRC blas/dg_op2_gemv_kernel.cu)
endif()

if(DIM EQUAL 2)
list(APPEND COMMON_SRC dg_constants/dg_constants_2d.cpp dg_mesh_op/dg_mesh_2d_op.cpp dg_operators_op/dg_operators_2d_op.cpp)
list(APPEND CPU_SRC dg_mesh/dg_mesh_2d_cpu.cpp)
list(APPEND GPU_SRC dg_mesh/dg_mesh_2d_gpu.cu)
elseif(DIM EQUAL 3)
list(APPEND COMMON_SRC dg_constants/dg_constants_3d.cpp dg_mesh_op/dg_mesh_3d_op.cpp dg_operators_op/dg_operators_3d_op.cpp)
list(APPEND COMMON_SRC dg_dat_pool.cpp)
list(APPEND CPU_SRC dg_constants/dg_constants_3d_cpu.cpp)
list(APPEND GPU_SRC dg_constants/dg_constants_3d_gpu.cu)
list(APPEND GPU_SRC dg_operators/custom_kernels/custom_grad_3d.cu)
list(APPEND GPU_SRC dg_operators/custom_kernels/custom_mass.cu)
endif()

# Only needs to be built once
if(DIM EQUAL 2)
  add_library(dgtoolkit STATIC ${TOOLKIT_SRC})
  target_link_libraries(dgtoolkit ${ARMA_LIB} ${OPENBLAS_LIB} ${EXTRA_LIBS})
  set(DG_LIBS dgtoolkit)
endif()

if(DIM EQUAL 2)
  set(DG_LIB_PREFIX op2dgtoolkit_2d)
elseif(DIM EQUAL 3)
  set(DG_LIB_PREFIX op2dgtoolkit_3d)
endif()

if(SOA EQUAL 1)
  set(DG_COMPILER_DEFS DG_OP2_SOA)
else()
  set(DG_COMPILER_DEFS "")
endif()

if(BUILD_SN AND BUILD_CPU)
  add_library(${DG_LIB_PREFIX}_seq STATIC ${COMMON_SRC} ${CPU_SRC} seq/dg_tookit_seqkernels.cpp)
  target_link_libraries(${DG_LIB_PREFIX}_seq dgtoolkit -L${OP2_DIR}/lib -lop2_seq -lop2_hdf5 ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} ${EXTRA_LIBS})
  target_compile_definitions(${DG_LIB_PREFIX}_seq PRIVATE ${LIBXSMM_COMPILE_OPTS} DG_ORDER=${ORDER} DG_DIM=${DIM} DG_COL_MAJ MAX_CONST_SIZE=1024)

  add_library(${DG_LIB_PREFIX}_openmp STATIC ${COMMON_SRC} ${CPU_SRC} openmp/dg_tookit_kernels.cpp)
  target_link_libraries(${DG_LIB_PREFIX}_openmp dgtoolkit -L${OP2_DIR}/lib -lop2_openmp -lop2_hdf5 ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} ${EXTRA_LIBS})
  target_compile_definitions(${DG_LIB_PREFIX}_openmp PRIVATE ${LIBXSMM_COMPILE_OPTS} DG_ORDER=${ORDER} DG_DIM=${DIM} DG_COL_MAJ MAX_CONST_SIZE=1024)

  list(APPEND DG_LIBS ${DG_LIB_PREFIX}_seq ${DG_LIB_PREFIX}_openmp)
endif()

if(BUILD_SN AND BUILD_GPU)
  add_library(${DG_LIB_PREFIX}_cuda STATIC ${COMMON_SRC} ${GPU_SRC} cuda/dg_tookit_kernels.cu)
  target_link_libraries(${DG_LIB_PREFIX}_cuda dgtoolkit -L${OP2_DIR}/lib -lop2_cuda -lop2_hdf5 ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} ${EXTRA_LIBS})
  target_compile_definitions(${DG_LIB_PREFIX}_cuda PRIVATE ${DG_COMPILER_DEFS} OP2_DG_CUDA DG_ORDER=${ORDER} DG_DIM=${DIM} DG_COL_MAJ MAX_CONST_SIZE=1024)

  set_property(TARGET ${DG_LIB_PREFIX}_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)

  list(APPEND DG_LIBS ${DG_LIB_PREFIX}_cuda)
endif()

if(BUILD_MPI AND BUILD_CPU)
  add_library(${DG_LIB_PREFIX}_mpi STATIC ${COMMON_SRC} ${CPU_SRC} seq/dg_tookit_seqkernels.cpp)
  target_link_libraries(${DG_LIB_PREFIX}_mpi dgtoolkit -L${OP2_DIR}/lib -lop2_mpi ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} ${PART_LIB} ${EXTRA_LIBS})
  target_compile_definitions(${DG_LIB_PREFIX}_mpi PRIVATE ${LIBXSMM_COMPILE_OPTS} DG_MPI DG_ORDER=${ORDER} DG_DIM=${DIM} DG_COL_MAJ MAX_CONST_SIZE=1024)

  add_library(${DG_LIB_PREFIX}_mpi_openmp STATIC ${COMMON_SRC} ${CPU_SRC} openmp/dg_tookit_kernels.cpp)
  target_link_libraries(${DG_LIB_PREFIX}_mpi_openmp dgtoolkit -L${OP2_DIR}/lib -lop2_mpi ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} ${PART_LIB} ${EXTRA_LIBS})
  target_compile_definitions(${DG_LIB_PREFIX}_mpi_openmp PRIVATE ${LIBXSMM_COMPILE_OPTS} DG_MPI DG_ORDER=${ORDER} DG_DIM=${DIM} DG_COL_MAJ MAX_CONST_SIZE=1024)

  list(APPEND DG_LIBS ${DG_LIB_PREFIX}_mpi ${DG_LIB_PREFIX}_mpi_openmp)
endif()

if(BUILD_MPI AND BUILD_GPU)
  add_library(${DG_LIB_PREFIX}_mpi_cuda STATIC ${COMMON_SRC} ${GPU_SRC} cuda/dg_tookit_kernels.cu)
  target_link_libraries(${DG_LIB_PREFIX}_mpi_cuda dgtoolkit -L${OP2_DIR}/lib -lop2_mpi_cuda ${HDF5_LIB} ${ARMA_LIB} ${OPENBLAS_LIB} ${PART_LIB} ${EXTRA_LIBS})
  target_compile_definitions(${DG_LIB_PREFIX}_mpi_cuda PRIVATE ${DG_COMPILER_DEFS} DG_MPI OP2_DG_CUDA DG_ORDER=${ORDER} DG_DIM=${DIM} DG_COL_MAJ MAX_CONST_SIZE=1024)

  set_property(TARGET ${DG_LIB_PREFIX}_mpi_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)

  list(APPEND DG_LIBS ${DG_LIB_PREFIX}_mpi_cuda)
endif()

install(TARGETS ${DG_LIBS} DESTINATION lib)
