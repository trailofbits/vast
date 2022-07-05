#
# Copyright (c) 2022-present, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

cmake_minimum_required(VERSION 3.21)

project(vast-example LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

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

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
add_definitions(${CLANG_DEFINITIONS})

#
# VAST
#
set(VAST_CMAKE_DIR "" CACHE PATH "VAST cmake config directory")

set(VAST_CMAKE_FILE "${VAST_CMAKE_DIR}/vastConfig.cmake")
if(NOT EXISTS "${VAST_CMAKE_FILE}")
  message(FATAL_ERROR " VAST_CMAKE_FILE (${VAST_CMAKE_FILE}) is invalid.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${VAST_CMAKE_DIR}")


#
# Example executable
#
add_executable(codegen codegen.cpp)

find_package(vast REQUIRED)

get_property(DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(codegen
    PRIVATE
        vast::vast

        MLIRIR

        ${MLIR_DIALECT_LIBS}
        ${CONVERSION_LIBS}

        clangAST
        clangASTMatchers
        clangBasic
        clangFrontend
        clangSerialization
        clangTooling
)

if (NOT LLVM_ENABLE_RTTI)
  set_target_properties(codegen PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

