set(KERNEL_HEADERS kernels/init_grid.h kernels/init_edges.h kernels/init_nodes.h)
list(APPEND KERNEL_HEADERS kernels/div.h kernels/curl.h kernels/grad.h)
list(APPEND KERNEL_HEADERS kernels/init_cubature.h kernels/init_gauss.h)
list(APPEND KERNEL_HEADERS kernels/cub_grad.h kernels/cub_div.h kernels/cub_grad_weak.h)
list(APPEND KERNEL_HEADERS kernels/cub_div_weak.h kernels/inv_J.h)

set(OPENBLAS_SRC openBLAS_op/init_grid_op.cpp openBLAS_op/op2_gemv_op.cpp openBLAS_op/op2_gemm_op.cpp)
list(APPEND OPENBLAS_SRC openBLAS_op/op2_gemv_batch_op.cpp openBLAS_op/op2_gemm_batch_op.cpp)
list(APPEND OPENBLAS_SRC openBLAS_op/init_gauss_op.cpp openBLAS_op/inv_op.cpp)

set(CUBLAS_SRC cuBLAS_op/init_grid_op.cu cuBLAS_op/op2_gemv_op.cu cuBLAS_op/op2_gemm_op.cu)
list(APPEND CUBLAS_SRC cuBLAS_op/op2_gemv_batch_op.cu cuBLAS_op/op2_gemm_batch_op.cu)
list(APPEND CUBLAS_SRC cuBLAS_op/init_gauss_op.cu cuBLAS_op/inv_op.cu)

set(COMMON_SRC dg_mesh_op.cpp dg_operators_op.cpp dg_constants_setup.cpp)
list(APPEND COMMON_SRC utils/misc.cpp utils/nodes.cpp utils/polynomial.cpp utils/vandermonde.cpp)
set(CPU_SRC dg_constants.cpp)
set(GPU_SRC dg_constants.cu)

op2_application(op2dgtoolkit_seq LIBS op2_seq SOURCES ${COMMON_SRC} ${CPU_SRC} ${OPENBLAS_SRC} seq/dg_mesh_seqkernels.cpp ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_seq PUBLIC ${ARMA_DIR}/include ${OPENBLAS_DIR}/include ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_seq -L${ARMA_DIR}/lib -larmadillo -L${OPENBLAS_DIR}/lib -lopenblas)
target_compile_definitions(op2dgtoolkit_seq PRIVATE DG_ORDER=3)

op2_application(op2dgtoolkit_openmp LIBS op2_openmp SOURCES ${COMMON_SRC} ${CPU_SRC} ${OPENBLAS_SRC} openmp/dg_mesh_kernels.cpp ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_openmp PUBLIC ${ARMA_DIR}/include ${OPENBLAS_DIR}/include ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_openmp -L${ARMA_DIR}/lib -larmadillo -L${OPENBLAS_DIR}/lib -lopenblas)
target_compile_definitions(op2dgtoolkit_openmp PRIVATE DG_ORDER=3)

op2_application(op2dgtoolkit_cuda LIBS op2_cuda SOURCES ${COMMON_SRC} ${GPU_SRC} ${CUBLAS_SRC} cuda/dg_mesh_kernels.cu ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_cuda PUBLIC ${ARMA_DIR}/include ${CUDAToolkit_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_cuda -L${ARMA_DIR}/lib -larmadillo -L${OPENBLAS_DIR}/lib -lopenblas CUDA::cudart_static CUDA::cublas)
set_property(TARGET op2dgtoolkit_cuda PROPERTY CUDA_ARCHITECTURES OFF)
target_compile_definitions(op2dgtoolkit_cuda PRIVATE OP2_DG_CUDA DG_ORDER=3)
#set_target_properties(op2dgtoolkit_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

op2_application(op2dgtoolkit_mpi LIBS op2_mpi SOURCES ${COMMON_SRC} ${CPU_SRC} ${OPENBLAS_SRC} seq/dg_mesh_seqkernels.cpp ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_mpi PUBLIC ${ARMA_DIR}/include ${OPENBLAS_DIR}/include ${PROJECT_SOURCE_DIR}/include ${PARMETIS_DIR}/include)
target_link_libraries(op2dgtoolkit_mpi -L${ARMA_DIR}/lib -larmadillo -L${OPENBLAS_DIR}/lib -lopenblas ${PARMETIS_DIR}/lib/libparmetis.a)
target_compile_definitions(op2dgtoolkit_mpi PRIVATE DG_ORDER=3)

op2_application(op2dgtoolkit_mpi_openmp LIBS op2_mpi op2_openmp SOURCES ${COMMON_SRC} ${CPU_SRC} ${OPENBLAS_SRC} openmp/dg_mesh_kernels.cpp ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_mpi_openmp PUBLIC ${ARMA_DIR}/include ${OPENBLAS_DIR}/include ${PROJECT_SOURCE_DIR}/include ${PARMETIS_DIR}/include)
target_link_libraries(op2dgtoolkit_mpi_openmp -L${ARMA_DIR}/lib -larmadillo -L${OPENBLAS_DIR}/lib -lopenblas ${PARMETIS_DIR}/lib/libparmetis.a)
target_compile_definitions(op2dgtoolkit_mpi_openmp PRIVATE DG_ORDER=3)

op2_application(op2dgtoolkit_mpi_cuda LIBS op2_mpi_cuda SOURCES ${COMMON_SRC} ${GPU_SRC} ${CUBLAS_SRC} cuda/dg_mesh_kernels.cu ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_mpi_cuda PUBLIC ${ARMA_DIR}/include ${CUDAToolkit_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_mpi_cuda -L${ARMA_DIR}/lib -larmadillo -L${OPENBLAS_DIR}/lib -lopenblas CUDA::cudart_static CUDA::cublas)
set_property(TARGET op2dgtoolkit_mpi_cuda PROPERTY CUDA_ARCHITECTURES OFF)
target_compile_definitions(op2dgtoolkit_mpi_cuda PRIVATE OP2_DG_CUDA DG_ORDER=3)
#set_target_properties(op2dgtoolkit_mpi_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(DG_LIBS op2dgtoolkit_seq op2dgtoolkit_openmp op2dgtoolkit_cuda op2dgtoolkit_mpi op2dgtoolkit_mpi_openmp op2dgtoolkit_mpi_cuda)

install(TARGETS ${DG_LIBS} DESTINATION lib)
