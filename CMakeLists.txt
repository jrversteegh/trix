cmake_minimum_required(VERSION 3.5)

project(libmtrx)

set (CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
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
  ${TRIX_INCLUDE_DIR}
)
add_compile_definitions(
  FMT_HEADER_ONLY=1
)

FILE(GLOB lib_sources src/*.cpp)

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
      trix_static
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
      trix_static
    )
    add_test(${NEW_BENCHMARK} benchmarks/${NEW_BENCHMARK})
  endmacro()

  new_test(test_matrix)
  new_benchmark(test_performance)
endif()
