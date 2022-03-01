#
# Copyright (c) 2021-present, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

cmake_minimum_required(VERSION 3.16)

# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

if (POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif ()


project(vast
    LANGUAGES C CXX
    VERSION 0.1.0
    DESCRIPTION "Verbose intermediate representation for program analysis"
    HOMEPAGE_URL "https://github.com/trailofbits/vast.git"
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


include(prevent_in_source_builds)

include(GNUInstallDirs)

# check if vast is being used directly or via add_subdirectory,
# but allow overriding
if(NOT DEFINED VAST_MASTER_PROJECT)
    if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(VAST_MASTER_PROJECT ON)
    else()
        set(VAST_MASTER_PROJECT OFF)
    endif()
endif()

include(project_settings)
include(project_utils)

#
# CCACHE
#
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

FindAndSelectClangCompiler()

#
# LLVM & MLIR & Clang
#
set(LLVM_INSTALL_DIR "" CACHE PATH "LLVM installation directory")

set(LLVM_INCLUDE_DIR "${LLVM_INSTALL_DIR}/include/llvm")
if(NOT EXISTS "${LLVM_INCLUDE_DIR}")
  message(FATAL_ERROR " LLVM_INSTALL_DIR (${LT_LLVM_INCLUDE_DIR}) is invalid.")
endif()

set(LLVM_CMAKE_FILE "${LLVM_INSTALL_DIR}/lib/cmake/llvm/LLVMConfig.cmake")
if(NOT EXISTS "${LLVM_CMAKE_FILE}")
  message(FATAL_ERROR " LLVM_CMAKE_FILE (${LLVM_CMAKE_FILE}) is invalid.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/llvm/")

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

#
# MLIR
#
set(MLIR_CMAKE_FILE "${LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
if(NOT EXISTS "${MLIR_CMAKE_FILE}")
  message(FATAL_ERROR " MLIR_CMAKE_FILE (${MLIR_CMAKE_FILE}) is invalid.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/mlir/")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

#
# Clang
#
set(CLANG_CMAKE_FILE "${LLVM_INSTALL_DIR}/lib/cmake/clang/ClangConfig.cmake")
if(NOT EXISTS "${CLANG_CMAKE_FILE}")
  message(FATAL_ERROR " CLANG_CMAKE_FILE (${CLANG_CMAKE_FILE}) is invalid.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/clang/")

find_package(Clang REQUIRED CONFIG)

#
# LLVM Libraries
#
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if (NOT LLVM_ENABLE_RTTI)
  set_target_properties(vast_settings PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

# TODO(Heno): clean up & modernize
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
add_definitions(${CLANG_DEFINITIONS})

#
# VAST build settings
#

add_library(vast_settings INTERFACE)

include(cmake/compiler_warnings.cmake)
set_project_warnings(vast_settings)

target_include_directories(vast_settings INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
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
add_subdirectory(tools)

add_library(vast INTERFACE)

target_link_libraries(vast
  INTERFACE
    vast_translation_api
    vast_settings
)

target_include_directories(vast
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_target_properties(vast
  PROPERTIES PUBLIC_HEADER "${VAST_TRANSLATION_HEADERS}"
)

# test options
option(ENABLE_TESTING "Enable Test Builds" ON)

if (ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

#
# install settings
#

option(VAST_INSTALL "Generate the install target." ${VAST_MASTER_PROJECT})

if (VAST_INSTALL)
  set(VAST_CMAKE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

  set(VAST_INSTALL_TARGETS
    vast
    vast_translation_api
    vast_settings
    MLIRHighLevel
  )

  set(VAST_EXPORT_NAME vastTargets)

  install(TARGETS ${VAST_INSTALL_TARGETS}
    EXPORT ${VAST_EXPORT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      NAMELINK_SKIP
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(EXPORT ${VAST_EXPORT_NAME}
    FILE ${VAST_EXPORT_NAME}.cmake
    NAMESPACE vast::
    DESTINATION ${VAST_CMAKE_INSTALL_DIR}
  )

  install(TARGETS ${VAST_INSTALL_TARGETS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      NAMELINK_ONLY
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()
