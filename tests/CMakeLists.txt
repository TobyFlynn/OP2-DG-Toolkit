# Only need to compile one version for dg_utils_tests as it always runs on CPU not matter what OP2 code gen is used
set(TEST_SRC dg_utils_tests.cpp dg_utils/misc.cpp dg_utils/polynomial.cpp dg_utils/vandermonde.cpp dg_utils/nodes.cpp dg_utils/matrices.cpp)
op2_application(dg_utils_tests LIBS op2_openmp SOURCES ${TEST_SRC})
target_include_directories(dg_utils_tests PUBLIC ${ARMA_DIR}/include ${OPENBLAS_DIR}/include ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(dg_utils_tests op2dgtoolkit_openmp -L${ARMA_DIR}/lib -larmadillo -L${OPENBLAS_DIR}/lib -lopenblas)
target_compile_definitions(dg_utils_tests PRIVATE DG_ORDER=3)
