set(KERNEL_HEADERS kernels/init_grid.h kernels/init_edges.h kernels/init_nodes.h)
list(APPEND KERNEL_HEADERS kernels/div.h kernels/curl.h kernels/grad.h)
list(APPEND KERNEL_HEADERS kernels/init_cubature.h kernels/init_gauss.h)

set(OPENBLAS_SRC openBLAS_op/init_grid_op.cpp openBLAS_op/op2_gemv_op.cpp openBLAS_op/op2_gemm_op.cpp)
list(APPEND OPENBLAS_SRC openBLAS_op/op2_gemv_batch_op.cpp openBLAS_op/op2_gemm_batch_op.cpp)
list(APPEND OPENBLAS_SRC openBLAS_op/init_gauss_op.cpp)

set(CUBLAS_SRC cuBLAS_op/init_grid_op.cu cuBLAS_op/op2_gemv_op.cu cuBLAS_op/op2_gemm_op.cu)
list(APPEND CUBLAS_SRC cuBLAS_op/op2_gemv_batch_op.cu cuBLAS_op/op2_gemm_batch_op.cu)
list(APPEND CUBLAS_SRC cuBLAS_op/init_gauss_op.cu)

set(COMMON_SRC dg_mesh_op.cpp dg_operators_op.cpp)
set(CPU_SRC dg_constants.cpp)
set(GPU_SRC dg_constants.cu)

op2_application(op2dgtoolkit_seq LIBS op2_seq op2_hdf5 SOURCES ${CPU_SRC} ${OPENBLAS_SRC} seq/dg_mesh_seqkernels.cpp ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_seq PUBLIC ${OPENBLAS_DIR}/include ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_seq -L${OPENBLAS_DIR}/lib -lopenblas)

op2_application(op2dgtoolkit_openmp LIBS op2_openmp op2_hdf5 SOURCES ${CPU_SRC} ${OPENBLAS_SRC} openmp/dg_mesh_kernels.cpp ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_openmp PUBLIC ${OPENBLAS_DIR}/include ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_openmp -L${OPENBLAS_DIR}/lib -lopenblas)

op2_application(op2dgtoolkit_cuda LIBS op2_cuda op2_hdf5 SOURCES ${GPU_SRC} ${CUBLAS_SRC} cuda/dg_mesh_kernels.cu ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_cuda PUBLIC ${CUDAToolkit_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_cuda CUDA::cudart_static CUDA::cublas)
set_property(TARGET op2dgtoolkit_cuda PROPERTY CUDA_ARCHITECTURES OFF)
target_compile_definitions(op2dgtoolkit_cuda PRIVATE OP2_DG_CUDA)

op2_application(op2dgtoolkit_mpi LIBS op2_mpi op2_hdf5 SOURCES ${CPU_SRC} ${OPENBLAS_SRC} seq/dg_mesh_seqkernels.cpp ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_mpi PUBLIC ${OPENBLAS_DIR}/include ${PROJECT_SOURCE_DIR}/include ${PARMETIS_DIR}/include)
target_link_libraries(op2dgtoolkit_mpi ${PARMETIS_DIR}/lib/libparmetis.a -L${OPENBLAS_DIR}/lib -lopenblas)

op2_application(op2dgtoolkit_mpi_openmp LIBS op2_mpi op2_openmp op2_hdf5 SOURCES ${CPU_SRC} ${OPENBLAS_SRC} openmp/dg_mesh_kernels.cpp ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_mpi_openmp PUBLIC ${OPENBLAS_DIR}/include ${PROJECT_SOURCE_DIR}/include ${PARMETIS_DIR}/include)
target_link_libraries(op2dgtoolkit_mpi_openmp ${PARMETIS_DIR}/lib/libparmetis.a -L${OPENBLAS_DIR}/lib -lopenblas)

op2_application(op2dgtoolkit_mpi_cuda LIBS op2_mpi_cuda op2_hdf5 SOURCES ${GPU_SRC} ${CUBLAS_SRC} cuda/dg_mesh_kernels.cu ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_mpi_cuda PUBLIC ${CUDAToolkit_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_mpi_cuda CUDA::cudart_static CUDA::cublas)
set_property(TARGET op2dgtoolkit_mpi_cuda PROPERTY CUDA_ARCHITECTURES OFF)
target_compile_definitions(op2dgtoolkit_mpi_cuda PRIVATE OP2_DG_CUDA)

set(DG_LIBS op2dgtoolkit_seq op2dgtoolkit_openmp op2dgtoolkit_cuda op2dgtoolkit_mpi op2dgtoolkit_mpi_openmp op2dgtoolkit_mpi_cuda)

install(TARGETS ${DG_LIBS} DESTINATION lib)
