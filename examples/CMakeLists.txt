file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/examples/*.cpp")


find_package(MPI)
if (MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
    set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    set(CMAKE_C_COMPILER mpicc)
    set(CMAKE_CXX_COMPILER mpicxx)
else()
    message(FATAL_ERROR "Could not find any MPI installation!!!")
endif()


foreach (SRCFILE IN LISTS cpu_source_files)
    get_filename_component(EXE ${SRCFILE} NAME_WLE)
    message(STATUS "Target = ${EXE}")
    add_executable(${EXE} ${EXE}.cpp)
    target_link_libraries(${EXE} PRIVATE SZ3 csv ${MPI_C_LIBRARIES})
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${EXE}$<TARGET_FILE_SUFFIX:${EXE}> TYPE BIN)

endforeach ()

install(FILES ${PROJECT_SOURCE_DIR}/examples/testfloat_8_8_128.dat DESTINATION ${CMAKE_INSTALL_DATADIR}/SZ3)
