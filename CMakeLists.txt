cmake_minimum_required(VERSION 3.1)
project(tsne)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wpedantic -O3 \
    -ffast-math -funroll-loops")

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/cell.cpp
    ${CMAKE_SOURCE_DIR}/src/datapoint.cpp
    ${CMAKE_SOURCE_DIR}/src/sptree.cpp
    ${CMAKE_SOURCE_DIR}/src/tsne.cpp
    ${CMAKE_SOURCE_DIR}/src/vptree.cpp
)

find_package(OpenMP 2)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "Couldn't find OpenMP. Building without...")
endif()

add_library(tsne MODULE ${SOURCES})
target_include_directories(tsne PRIVATE ${INCLUDE_DIR})
set_property(TARGET tsne PROPERTY POSITION_INDEPENDENT_CODE ON)

