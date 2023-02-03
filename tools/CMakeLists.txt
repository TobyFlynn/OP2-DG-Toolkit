find_package(VTK)
if(NOT VTK_FOUND)
  message(FATAL_ERROR "VTK could not be found")
endif()

add_executable(vtk2hdf5_2D vtk2hdf5_2D.cpp)
target_link_libraries(vtk2hdf5_2D -L${OP2_DIR}/lib -lop2_mpi -lop2_hdf5 ${PART_LIB} ${MPI_CXX_LIBRARIES} ${VTK_LIBRARIES} ${HDF5_LIB} ${EXTRA_LIBS})
target_include_directories(vtk2hdf5_2D PRIVATE ${VTK_USE_FILE})
target_compile_definitions(vtk2hdf5_2D PRIVATE OP2_PARTITIONER=${PART_LIB_NAME})

add_executable(vtk2hdf5_3D vtk2hdf5_3D.cpp)
target_link_libraries(vtk2hdf5_3D -L${OP2_DIR}/lib -lop2_mpi -lop2_hdf5 ${PART_LIB} ${MPI_CXX_LIBRARIES} ${VTK_LIBRARIES} ${HDF5_LIB} ${EXTRA_LIBS})
target_include_directories(vtk2hdf5_3D PRIVATE ${VTK_USE_FILE})
target_compile_definitions(vtk2hdf5_3D PRIVATE OP2_PARTITIONER=${PART_LIB_NAME})

add_executable(vtk2hdf5_periodic_cube_3D vtk2hdf5_periodic_cube_3D.cpp)
target_link_libraries(vtk2hdf5_periodic_cube_3D -L${OP2_DIR}/lib -lop2_mpi -lop2_hdf5 ${PART_LIB} ${MPI_CXX_LIBRARIES} ${VTK_LIBRARIES} ${HDF5_LIB} ${EXTRA_LIBS})
target_include_directories(vtk2hdf5_periodic_cube_3D PRIVATE ${VTK_USE_FILE})
target_compile_definitions(vtk2hdf5_periodic_cube_3D PRIVATE OP2_PARTITIONER=${PART_LIB_NAME})

if(NOT HIGHFIVE_DIR)
  message(FATAL_ERROR "HIGHFIVE_DIR is not specified")
endif()

add_executable(hdf52vtk_2D hdf52vtk_2D.cpp)
target_link_libraries(hdf52vtk_2D ${VTK_LIBRARIES} ${HDF5_LIB} ${EXTRA_LIBS})
target_include_directories(hdf52vtk_2D PRIVATE ${VTK_USE_FILE} ${HIGHFIVE_DIR})
target_compile_definitions(hdf52vtk_2D PRIVATE DG_ORDER=${ORDER} DG_DIM=2)

add_executable(hdf52vtk_3D hdf52vtk_3D.cpp)
target_link_libraries(hdf52vtk_3D -L${OP2_DIR}/lib -lop2_seq -lop2_hdf5 ${VTK_LIBRARIES} ${HDF5_LIB} ${EXTRA_LIBS})
target_include_directories(hdf52vtk_3D PRIVATE ${VTK_USE_FILE} ${HIGHFIVE_DIR})
target_compile_definitions(hdf52vtk_3D PRIVATE DG_ORDER=${ORDER} DG_DIM=3)