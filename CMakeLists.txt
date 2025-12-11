cmake_minimum_required(VERSION 3.0...3.26)

set(CMAKE_BUILD_TYPE Debug)

# set(CMAKE_BUILD_TYPE Release)

project(mergegraph
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

add_library(mergegraph INTERFACE)
add_library(mergegraph::mergegraph ALIAS mergegraph)

target_include_directories(mergegraph INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mergegraph
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS mergegraph
    EXPORT mergegraphTargets)

install(EXPORT mergegraphTargets
    FILE mergegraphConfig.cmake
    NAMESPACE mergegraph::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mergegraph)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(COMPILER_SUPPORTS_AVX512)
        add_definitions(-DUSE_AVX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
        message("AVX512 enabled")
    else()
        check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
        if(COMPILER_SUPPORTS_AVX)
            add_definitions(-DUSE_AVX)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
            message("AVX enabled")
        else()
            add_definitions(-DUSE_SSE)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
            message("SSE enabled")
        endif()
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_definitions(-DUSE_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
    message("AVX512 enabled for MSVC")
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(mergegraph_EXAMPLES "Build examples and tests." ON)
else()
    option(mergegraph_EXAMPLES "Build examples and tests." OFF)
endif()
if(mergegraph_EXAMPLES)
    set(CMAKE_CXX_STANDARD 17)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      SET( CMAKE_CXX_FLAGS  "-Ofast -std=c++20 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" )
      check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
      if(COMPILER_SUPPORT_NATIVE_FLAG)
        SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
        message("set -march=native flag")
      else()
        check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
        if(COMPILER_SUPPORT_M1_FLAG)
          SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
          message("set -mcpu=apple-m1 flag")
        endif()
      endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      SET( CMAKE_CXX_FLAGS  "-mavx2 -Ofast -lrt -std=c++20 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" )
    endif()

    add_executable(merge_test examples/cpp/merge_test.cpp)
    target_link_libraries(merge_test mergegraph)
    add_executable(vamana_build examples/cpp/vamana_build.cpp)
    target_link_libraries(vamana_build mergegraph)
    add_executable(merge_NND_streaming examples/cpp/merge_NND_streaming.cpp)
    target_link_libraries(merge_NND_streaming mergegraph)

endif()