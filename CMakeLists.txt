cmake_minimum_required(VERSION 3.1)
project(tsne_cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -Werror -Wno-unknown-pragmas")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -funroll-loops")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

# default to Release if not set explicitily
#if("${CMAKE_BUILD_TYPE}" STREQUAL "")
#  set(CMAKE_BUILD_TYPE Release)
#  message(STATUS "Using Release as default build type.")
#endif()

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/cell.cpp
    ${CMAKE_SOURCE_DIR}/src/datapoint.cpp
    ${CMAKE_SOURCE_DIR}/src/sptree.cpp
    ${CMAKE_SOURCE_DIR}/src/tsne.cpp
    ${CMAKE_SOURCE_DIR}/src/vptree.cpp
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

#find_package(PythonLibs 2.7 REQUIRED)
#find_package(PythonInterp 2.7 REQUIRED) # needed for numpy find module
#find_package(Boost REQUIRED python27) # TODO: add version for boost
#find_package(NumPy REQUIRED) # TODO: add version for numpy
#
#add_library(${CMAKE_PROJECT_NAME} SHARED ${PYTHON_SOURCES} ${SOURCES})
#target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${INCLUDE_DIR}
#    ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
#target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
#set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
#set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY PREFIX "")

add_executable(main ${CPP_MAIN_SOURCES} ${SOURCES})
target_include_directories(main PUBLIC ${INCLUDE_DIR})
