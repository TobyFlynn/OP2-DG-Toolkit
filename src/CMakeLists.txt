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

set(COMMON_SRC dg_mesh_op.cpp dg_operators_op.cpp)
set(CPU_SRC dg_constants.cpp)
set(GPU_SRC dg_constants.cu)

op2_application(op2dgtoolkit_seq LIBS op2_seq op2_hdf5 SOURCES ${COMMON_SRC} ${CPU_SRC} ${OPENBLAS_SRC} seq/dg_mesh_seqkernels.cpp ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_seq PUBLIC ${ARMA_DIR}/include ${OpenBLAS_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_seq libhdf5.so -L${ARMA_DIR}/lib -larmadillo ${OpenBLAS_LIBRARIES})

op2_application(op2dgtoolkit_openmp LIBS op2_openmp op2_hdf5 SOURCES ${COMMON_SRC} ${CPU_SRC} ${OPENBLAS_SRC} openmp/dg_mesh_kernels.cpp ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_openmp PUBLIC ${ARMA_DIR}/include ${OpenBLAS_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_openmp libhdf5.so -L${ARMA_DIR}/lib -larmadillo ${OpenBLAS_LIBRARIES})

op2_application(op2dgtoolkit_cuda LIBS op2_cuda op2_hdf5 SOURCES ${COMMON_SRC} ${GPU_SRC} ${CUBLAS_SRC} cuda/dg_mesh_kernels.cu ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_cuda PUBLIC ${ARMA_DIR}/include ${CUDAToolkit_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_cuda libhdf5.so -L${ARMA_DIR}/lib -larmadillo ${OpenBLAS_LIBRARIES} CUDA::cudart_static CUDA::cublas)
set_property(TARGET op2dgtoolkit_cuda PROPERTY CUDA_ARCHITECTURES OFF)
target_compile_definitions(op2dgtoolkit_cuda PRIVATE OP2_DG_CUDA)
#set_target_properties(op2dgtoolkit_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

op2_application(op2dgtoolkit_mpi LIBS op2_mpi op2_hdf5 SOURCES ${COMMON_SRC} ${CPU_SRC} ${OPENBLAS_SRC} seq/dg_mesh_seqkernels.cpp ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_mpi PUBLIC ${ARMA_DIR}/include ${OpenBLAS_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_mpi libhdf5.so -L${ARMA_DIR}/lib -larmadillo ${OpenBLAS_LIBRARIES} libptscotch.a libscotch.a)

op2_application(op2dgtoolkit_mpi_openmp LIBS op2_mpi op2_openmp op2_hdf5 SOURCES ${COMMON_SRC} ${CPU_SRC} ${OPENBLAS_SRC} openmp/dg_mesh_kernels.cpp ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_mpi_openmp PUBLIC ${ARMA_DIR}/include ${OpenBLAS_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_mpi_openmp libhdf5.so -L${ARMA_DIR}/lib -larmadillo ${OpenBLAS_LIBRARIES} libptscotch.a libscotch.a)

op2_application(op2dgtoolkit_mpi_cuda LIBS op2_mpi_cuda op2_hdf5 SOURCES ${COMMON_SRC} ${GPU_SRC} ${CUBLAS_SRC} cuda/dg_mesh_kernels.cu ${KERNEL_HEADERS} BIN library)
target_include_directories(op2dgtoolkit_mpi_cuda PUBLIC ${ARMA_DIR}/include ${CUDAToolkit_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_mpi_cuda libhdf5.so -L${ARMA_DIR}/lib -larmadillo ${OpenBLAS_LIBRARIES} CUDA::cudart_static CUDA::cublas libptscotch.a libscotch.a)
set_property(TARGET op2dgtoolkit_mpi_cuda PROPERTY CUDA_ARCHITECTURES OFF)
target_compile_definitions(op2dgtoolkit_mpi_cuda PRIVATE OP2_DG_CUDA)
#set_target_properties(op2dgtoolkit_mpi_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(DG_LIBS op2dgtoolkit_seq op2dgtoolkit_openmp op2dgtoolkit_cuda op2dgtoolkit_mpi op2dgtoolkit_mpi_openmp op2dgtoolkit_mpi_cuda)

install(TARGETS ${DG_LIBS} DESTINATION lib)
