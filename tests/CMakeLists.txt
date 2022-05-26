# Only need to compile one version for dg_utils_tests as it always runs on CPU not matter what OP2 code gen is used
set(TEST_SRC dg_utils_tests.cpp dg_utils/misc.cpp dg_utils/polynomial.cpp dg_utils/vandermonde.cpp dg_utils/nodes.cpp dg_utils/matrices.cpp)

set(GFORT_LIB /gpfs/users/shared/apps/core/gcc/7.3.0/lib64/libgfortran.so.4)

add_executable(dg_utils_tests ${TEST_SRC})
target_link_libraries(dg_utils_tests dgtoolkit libhdf5.so -L${ARMA_DIR}/lib64 -larmadillo ${OpenBLAS_LIBRARIES} ${GFORT_LIB})
