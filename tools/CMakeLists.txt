find_package(VTK)
if(NOT VTK_FOUND)
  message(FATAL_ERROR "VTK could not be found")
endif()

add_executable(vtk2hdf5_2D vtk2hdf5_2D.cpp)
target_link_libraries(vtk2hdf5_2D -L${OP2_DIR}/lib -lop2_mpi -lop2_hdf5 ${PART_LIB} ${MPI_CXX_LIBRARIES} ${VTK_LIBRARIES} ${HDF5_LIB} ${EXTRA_LIBS})
target_include_directories(vtk2hdf5_2D PRIVATE ${VTK_USE_FILE})
target_compile_definitions(vtk2hdf5_2D PRIVATE OP2_PARTITIONER=${PART_LIB_NAME})
