# Copyright (c) 2022-present, Trail of Bits, Inc.

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

#
# MLIR
#
set(MLIR_CMAKE_FILE "${LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
if(NOT EXISTS "${MLIR_CMAKE_FILE}")
    message(FATAL_ERROR " MLIR_CMAKE_FILE (${MLIR_CMAKE_FILE}) is invalid.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/mlir/")

#
# Clang
#
set(CLANG_CMAKE_FILE "${LLVM_INSTALL_DIR}/lib/cmake/clang/ClangConfig.cmake")
if(NOT EXISTS "${CLANG_CMAKE_FILE}")
  message(FATAL_ERROR " CLANG_CMAKE_FILE (${CLANG_CMAKE_FILE}) is invalid.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/clang/")