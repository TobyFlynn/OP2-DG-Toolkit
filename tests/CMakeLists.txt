# Only need to compile one version for dg_utils_tests as it always runs on CPU not matter what OP2 code gen is used
set(TEST_SRC dg_utils_tests.cpp dg_utils/misc.cpp dg_utils/polynomial.cpp dg_utils/vandermonde.cpp dg_utils/nodes.cpp dg_utils/matrices.cpp)

add_executable(dg_utils_tests ${TEST_SRC})
target_link_libraries(dg_utils_tests dgtoolkit -L${ARMA_DIR}/lib -larmadillo -L${OPENBLAS_DIR}/lib -lopenblas)
