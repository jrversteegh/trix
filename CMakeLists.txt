cmake_minimum_required(VERSION 3.28)

project(trix)

if(NOT DEFINED ${CMAKE_BUILD_TYPE})
  set (CMAKE_BUILD_TYPE Release)
endif()


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_CXX_SCAN_FOR_MODULES TRUE)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmax-errors=1")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
    message(FATAL_ERROR "g++ 14 or new required. Set environment variable CXX to a newer g++")
  endif()
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=1")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19)
    message(FATAL_ERROR "clang++ 19 or new required. Set environment variable CXX to a newer clang++")
  endif()
endif()


if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
  set(TEST_LINK_FLAGS "")
  set(MODULE_OUTPUT_DIR ".")
else()
  set(TEST_LINK_FLAGS LINK_FLAGS "-Wl,--no-as-needed")
  set(MODULE_OUTPUT_DIR "${CMAKE_BUILD_TYPE}")
endif()

if(BUILD_PYTHON)
  find_package(pybind11 REQUIRED)
  execute_process(COMMAND python-config --includes OUTPUT_VARIABLE Python_INCLUDES)
endif()
if(BUILD_TESTS)
  find_package(Boost 1.88.0 REQUIRED
    system date_time filesystem log log_setup unit_test_framework regex
    program_options
  )
  find_package(benchmark REQUIRED)
endif()

find_package(MKL QUIET)
if (TARGET MKL::MKL)
  add_library(BLAS::BLAS ALIAS MKL::MKL)
  set(HAVE_MKL 1)
else()
  find_package(OpenBLAS REQUIRED)
  add_library(BLAS::BLAS ALIAS OpenBLAS::OpenBLAS)
  set(HAVE_OPENBLAS 1)
endif()
find_package(Eigen3 REQUIRED NO_MODULE)
if (TARGET Eigen3::Eigen)
  set(HAVE_EIGEN 1)
endif()

find_package(fmt 11.2.0 REQUIRED)

execute_process(COMMAND git rev-parse --short HEAD OUTPUT_VARIABLE GITREV)
set(TRIX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(TRIX_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set(VERSION_H_IN ${TRIX_INCLUDE_DIR}/trix/version.h.in)
set(VERSION_H ${TRIX_INCLUDE_DIR}/trix/version.h)
set(CONFIG_H_IN ${TRIX_INCLUDE_DIR}/trix/config.h.in)
set(CONFIG_H ${TRIX_INCLUDE_DIR}/trix/config.h)
configure_file(${VERSION_H_IN} ${VERSION_H} @ONLY)
configure_file(${CONFIG_H_IN} ${CONFIG_H} @ONLY)
if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
configure_file(${CMAKE_SOURCE_DIR}/cmake/CMakeSettings.json.in ${CMAKE_SOURCE_DIR}/CMakeSettings.json @ONLY)
endif()

include_directories(
  ${Boost_INCLUDE_DIR}
  ${pybind11_INCLUDE_DIR}
  ${fmt_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${TRIX_INCLUDE_DIR}
)
add_compile_definitions(
  FMT_HEADER_ONLY=1
)

FILE(GLOB lib_sources src/*.cpp)
FILE(GLOB module_sources src/*.cppm)

add_library(trix OBJECT ${lib_sources})
message("Module output directory: ${MODULE_OUTPUT_DIR}")
set_target_properties(trix PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${MODULE_OUTPUT_DIR})
if(BUILD_SHARED)
  set_property(TARGET trix PROPERTY POSITION_INDEPENDENT_CODE ON)
  add_library(trix_shared SHARED $<TARGET_OBJECTS:trix>)
  set_target_properties(trix_shared PROPERTIES OUTPUT_NAME trix)
endif()
add_library(trix_static STATIC $<TARGET_OBJECTS:trix>)
set_target_properties(trix_static PROPERTIES OUTPUT_NAME trix)
if(BUILD_PYTHON)
  pybind11_add_module(trixx MODULE ${TRIX_SOURCE_DIR}/trixx/binding.cpp ${lib_sources})
endif()


enable_testing()

if(BUILD_TESTS)
  if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
      include(CTest)
  endif()

  macro(new_test NEW_TEST)
    set(TEST_SOURCES ${ARGN})
    list(TRANSFORM TEST_SOURCES PREPEND ${TRIX_SOURCE_DIR}/ )
    add_executable(${NEW_TEST}
      ${TEST_DIR}/${NEW_TEST}.cpp
      ${TEST_SOURCES}
    )
    target_include_directories(${NEW_TEST} PRIVATE
      ${TRIX_INCLUDE_DIR}
      ${TRIX_SOURCE_DIR}
    )
    set_target_properties(${NEW_TEST} PROPERTIES
      ${TEST_LINK_FLAGS}
      RUNTIME_OUTPUT_DIRECTORY tests
    )
    target_link_libraries(${NEW_TEST} PRIVATE
      Boost::system
      Boost::filesystem
      Boost::program_options
      Boost::log
      Boost::log_setup
      Boost::unit_test_framework
      BLAS::BLAS
      trix_static
    )
    target_sources(${NEW_TEST} PRIVATE 
      FILE_SET cxx_modules TYPE CXX_MODULES
      FILES ${module_sources}
    )
    add_test(${NEW_TEST} tests/${NEW_TEST})
  endmacro()

  macro(new_benchmark NEW_BENCHMARK)
    set(BENCHMARK_SOURCES ${ARGN})
    list(TRANSFORM BENCHMARK_SOURCES PREPEND ${TRIX_SOURCE_DIR}/ )
    add_executable(${NEW_BENCHMARK}
      ${TEST_DIR}/${NEW_BENCHMARK}.cpp
      ${BENCHMARK_SOURCES}
    )
    target_include_directories(${NEW_BENCHMARK} PRIVATE
      ${TRIX_INCLUDE_DIR}
      ${TRIX_SOURCE_DIR}
    )
    set_target_properties(${NEW_BENCHMARK} PROPERTIES
      ${BENCHMARK_LINK_FLAGS}
      RUNTIME_OUTPUT_DIRECTORY benchmarks
    )
    target_link_libraries(${NEW_BENCHMARK} PRIVATE
      benchmark::benchmark_main
      Boost::system
      Boost::filesystem
      Boost::program_options
      Boost::log
      Boost::log_setup
      Boost::unit_test_framework
      BLAS::BLAS
      trix_static
    )
    add_test(${NEW_BENCHMARK} benchmarks/${NEW_BENCHMARK})
  endmacro()

  new_test(test_vector)
  new_test(test_matrix)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 15)
    new_test(test_module)
  endif()
  new_benchmark(test_performance)
endif()
