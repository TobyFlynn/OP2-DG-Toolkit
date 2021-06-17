set(CPU_SRC constants.cpp)
set(GPU_SRC constants.cu)

op2_application(op2dgtoolkit_seq LIBS op2_seq op2_hdf5 SOURCES ${CPU_SRC} BIN library)
target_include_directories(op2dgtoolkit_seq PUBLIC ${OPENBLAS_DIR}/include ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_seq -L${OPENBLAS_DIR}/lib -lopenblas)

op2_application(op2dgtoolkit_openmp LIBS op2_openmp op2_hdf5 SOURCES ${CPU_SRC} BIN library)
target_include_directories(op2dgtoolkit_openmp PUBLIC ${OPENBLAS_DIR}/include ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_openmp -L${OPENBLAS_DIR}/lib -lopenblas)

op2_application(op2dgtoolkit_cuda LIBS op2_cuda op2_hdf5 SOURCES ${GPU_SRC} BIN library)
target_include_directories(op2dgtoolkit_cuda PUBLIC ${CUDAToolkit_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_cuda CUDA::cudart_static CUDA::cublas)
set_property(TARGET op2dgtoolkit_cuda PROPERTY CUDA_ARCHITECTURES OFF)
target_compile_definitions(op2dgtoolkit_cuda PRIVATE OP2_DG_CUDA)

op2_application(op2dgtoolkit_mpi LIBS op2_mpi op2_hdf5 SOURCES ${CPU_SRC} BIN library)
target_include_directories(op2dgtoolkit_mpi PUBLIC ${OPENBLAS_DIR}/include ${PROJECT_SOURCE_DIR}/include ${PARMETIS_DIR}/include)
target_link_libraries(op2dgtoolkit_mpi ${PARMETIS_DIR}/lib/libparmetis.a -L${OPENBLAS_DIR}/lib -lopenblas)

op2_application(op2dgtoolkit_mpi_openmp LIBS op2_mpi op2_openmp op2_hdf5 SOURCES ${CPU_SRC} BIN library)
target_include_directories(op2dgtoolkit_mpi_openmp PUBLIC ${OPENBLAS_DIR}/include ${PROJECT_SOURCE_DIR}/include ${PARMETIS_DIR}/include)
target_link_libraries(op2dgtoolkit_mpi_openmp ${PARMETIS_DIR}/lib/libparmetis.a -L${OPENBLAS_DIR}/lib -lopenblas)

op2_application(op2dgtoolkit_mpi_cuda LIBS op2_mpi_cuda op2_hdf5 SOURCES ${GPU_SRC} BIN library)
target_include_directories(op2dgtoolkit_mpi_cuda PUBLIC ${CUDAToolkit_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op2dgtoolkit_mpi_cuda CUDA::cudart_static CUDA::cublas)
set_property(TARGET op2dgtoolkit_mpi_cuda PROPERTY CUDA_ARCHITECTURES OFF)
target_compile_definitions(op2dgtoolkit_mpi_cuda PRIVATE OP2_DG_CUDA)
