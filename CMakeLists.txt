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
# LLVM
#

find_package(LLVM REQUIRED CONFIG)

#
# MLIR
#

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

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
# VAST librarries
#
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
