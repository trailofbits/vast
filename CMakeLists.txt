#
# Copyright (c) 2021-present, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

cmake_minimum_required(VERSION 3.18)

# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

if (POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif ()

include(cmake/vcpkg_helper.cmake)

project(vast
    LANGUAGES C CXX
    VERSION 0.0.0
    DESCRIPTION "Verbose intermediate representation for program analysis"
    HOMEPAGE_URL "https://github.com/trailofbits/vast.git"
)

include(cmake/prevent_in_source_builds.cmake)

include(GNUInstallDirs)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/settings.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils.cmake")

FindAndSelectClangCompiler()

#
# LLVM & MLIR
#

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# TODO(Heno): clean up & modernize
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#
# google log module
#
find_package(glog CONFIG QUIET)
if (NOT glog_FOUND)
  find_package(glog REQUIRED)
  list(APPEND PROJECT_LIBRARIES glog)
else()
  list(APPEND PROJECT_LIBRARIES glog::glog)
endif()

#
# gflags
#

find_package(gflags CONFIG QUIET)
if (NOT gflags_FOUND)
  find_package(gflags REQUIRED)
endif()
list(APPEND PROJECT_LIBRARIES gflags)

#
# VAST build settings
#

add_library(vast_settings INTERFACE)

include(cmake/compiler_warnings.cmake)
set_project_warnings(vast_settings)

target_include_directories(vast_settings INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# sanitizer options if supported by compiler
include(cmake/sanitizers.cmake)
enable_sanitizers(vast_settings)

# allow for static analysis options
include(cmake/static_analyzers.cmake)

#
# VAST libraries
#
add_subdirectory(include/vast)
add_subdirectory(lib)

#
# VAST executables
#
add_subdirectory(bin)


# test options
option(ENABLE_TESTING "Enable Test Builds" ON)

if (ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

#
# install settings
#

if(DEFINED WIN32)
  set(install_folder "${CMAKE_INSTALL_PREFIX}/vast")
else()
  set(install_folder "${CMAKE_INSTALL_PREFIX}")
endif()
