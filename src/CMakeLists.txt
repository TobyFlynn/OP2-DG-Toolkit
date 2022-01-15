set(OPENBLAS_SRC openBLAS_op/init_grid_op.cpp openBLAS_op/init_gauss_op.cpp openBLAS_op/inv_op.cpp)
set(CUBLAS_SRC cuBLAS_op/init_grid_op.cu cuBLAS_op/init_gauss_op.cu cuBLAS_op/inv_op.cu)

set(COMMON_SRC dg_mesh_op.cpp dg_operators_op.cpp dg_op2_blas_op.cpp dg_constants_setup.cpp)
list(APPEND COMMON_SRC utils/misc.cpp utils/nodes.cpp utils/polynomial.cpp utils/vandermonde.cpp utils/matrices.cpp)
set(CPU_SRC dg_constants.cpp)
set(GPU_SRC dg_constants.cu)

add_library(op2dgtoolkit_seq SHARED ${COMMON_SRC} ${CPU_SRC} ${OPENBLAS_SRC} seq/dg_mesh_seqkernels.cpp)
target_link_libraries(op2dgtoolkit_seq -L${OP2_DIR}/lib -lop2_seq -L${ARMA_DIR}/lib -larmadillo -L${OPENBLAS_DIR}/lib -lopenblas)
target_compile_definitions(op2dgtoolkit_seq PRIVATE DG_ORDER=3)

add_library(op2dgtoolkit_openmp SHARED ${COMMON_SRC} ${CPU_SRC} ${OPENBLAS_SRC} openmp/dg_mesh_kernels.cpp)
target_link_libraries(op2dgtoolkit_openmp -L${OP2_DIR}/lib -lop2_openmp -L${ARMA_DIR}/lib -larmadillo -L${OPENBLAS_DIR}/lib -lopenblas)
target_compile_definitions(op2dgtoolkit_openmp PRIVATE DG_ORDER=3)

add_library(op2dgtoolkit_cuda SHARED ${COMMON_SRC} ${GPU_SRC} ${CUBLAS_SRC} cuda/dg_mesh_kernels.cu)
target_link_libraries(op2dgtoolkit_cuda -L${OP2_DIR}/lib -lop2_cuda -L${ARMA_DIR}/lib -larmadillo -L${OPENBLAS_DIR}/lib -lopenblas CUDA::cudart_static CUDA::cublas)
set_property(TARGET op2dgtoolkit_cuda PROPERTY CUDA_ARCHITECTURES OFF)
target_compile_definitions(op2dgtoolkit_cuda PRIVATE OP2_DG_CUDA DG_ORDER=3)

add_library(op2dgtoolkit_mpi SHARED ${COMMON_SRC} ${CPU_SRC} ${OPENBLAS_SRC} seq/dg_mesh_seqkernels.cpp)
target_link_libraries(op2dgtoolkit_mpi -L${OP2_DIR}/lib -lop2_mpi -L${ARMA_DIR}/lib -larmadillo -L${OPENBLAS_DIR}/lib -lopenblas ${PARMETIS_DIR}/lib/libparmetis.a)
target_compile_definitions(op2dgtoolkit_mpi PRIVATE DG_ORDER=3)

add_library(op2dgtoolkit_mpi_openmp SHARED ${COMMON_SRC} ${CPU_SRC} ${OPENBLAS_SRC} openmp/dg_mesh_kernels.cpp)
target_link_libraries(op2dgtoolkit_mpi_openmp -L${OP2_DIR}/lib -lop2_mpi -lop2_openmp -L${ARMA_DIR}/lib -larmadillo -L${OPENBLAS_DIR}/lib -lopenblas ${PARMETIS_DIR}/lib/libparmetis.a)
target_compile_definitions(op2dgtoolkit_mpi_openmp PRIVATE DG_ORDER=3)

add_library(op2dgtoolkit_mpi_cuda SHARED ${COMMON_SRC} ${GPU_SRC} ${CUBLAS_SRC} cuda/dg_mesh_kernels.cu)
target_link_libraries(op2dgtoolkit_mpi_cuda -L${OP2_DIR}/lib -lop2_mpi_cuda -L${ARMA_DIR}/lib -larmadillo -L${OPENBLAS_DIR}/lib -lopenblas CUDA::cudart_static CUDA::cublas)
set_property(TARGET op2dgtoolkit_mpi_cuda PROPERTY CUDA_ARCHITECTURES OFF)
target_compile_definitions(op2dgtoolkit_mpi_cuda PRIVATE OP2_DG_CUDA DG_ORDER=3)

set(DG_LIBS op2dgtoolkit_seq op2dgtoolkit_openmp op2dgtoolkit_cuda op2dgtoolkit_mpi op2dgtoolkit_mpi_openmp op2dgtoolkit_mpi_cuda)

install(TARGETS ${DG_LIBS} DESTINATION lib)
