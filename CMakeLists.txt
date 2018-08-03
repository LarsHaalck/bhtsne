cmake_minimum_required(VERSION 3.5.2)
project(tsne_module)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -Werror -Wno-unknown-pragmas")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -funroll-loops")

# default to Release if not set explicitily
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Using Release as default build type.")
endif()

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/cell.cpp
    ${CMAKE_SOURCE_DIR}/src/datapoint.cpp
    ${CMAKE_SOURCE_DIR}/src/sptree.cpp
    ${CMAKE_SOURCE_DIR}/src/vptree.cpp
    ${CMAKE_SOURCE_DIR}/src/tsne.cpp
)
set(PYTHON_SOURCES
    ${CMAKE_SOURCE_DIR}/src/python_wrapper.cpp
)

set(CPP_MAIN_SOURCES
    ${CMAKE_SOURCE_DIR}/src/main.cpp
)


find_package(OpenMP 2)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "Couldn't find OpenMP. Building without...")
endif()

# TODO: use CMakes Python3-Module (v3.12, when Ubuntu supports it in a few years ;) )
find_package(PythonLibs REQUIRED)
find_package(Boost REQUIRED python36 numpy36)

add_library(${CMAKE_PROJECT_NAME} SHARED ${PYTHON_SOURCES} ${SOURCES})
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIRS})
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC Boost::python36 Boost::numpy36
    ${PYTHON_LIBRARIES})


set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY PREFIX "")

add_executable(main ${CPP_MAIN_SOURCES} ${SOURCES})
target_include_directories(main PUBLIC ${INCLUDE_DIR})
