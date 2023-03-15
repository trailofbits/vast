#
# Copyright (c) 2021-present, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

cmake_minimum_required(VERSION 3.21)

# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

if (POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(ENABLE_VCPKG_CONFIG "Enable vcpkg" OFF)

if (ENABLE_VCPKG_CONFIG)
  include(vast_vcpkg_helper)
endif()

project(VAST
    LANGUAGES C CXX
    VERSION 0.0.0
    DESCRIPTION "Verbose intermediate representation for program analysis"
    HOMEPAGE_URL "https://github.com/trailofbits/vast.git"
)

include(prevent_in_source_builds)

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
# Setup package version
#
setup_package_version_variables(vast)

set(VAST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(VAST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(VAST_INCLUDE_DIR ${VAST_BINARY_DIR}/include)

# Configure Vast Version.inc file.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/vast/Version.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/vast/Version.inc
)

# Configure Vast version info header file.
configure_file(
  ${VAST_SOURCE_DIR}/include/vast/Config/config.h.cmake
  ${VAST_BINARY_DIR}/include/vast/Config/config.h
)

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
find_package(LLVM 15 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

#
# MLIR
#
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

#
# Clang
#
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

# TODO(Heno): clean up & modernize
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
add_definitions(${CLANG_DEFINITIONS})

llvm_map_components_to_libnames(LLVM_LIBS
  ${LLVM_TARGETS_TO_BUILD} support target option
)

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

if (NOT LLVM_ENABLE_RTTI)
  message( FATAL_ERROR
    "VAST does not support build without RTTI, yet. Please build LLVM with LLVM_ENABLE_RTTI."
  )
endif()

# sanitizer options if supported by compiler
include(cmake/sanitizers.cmake)
enable_sanitizers(vast_settings)

# allow for static analysis options
include(cmake/static_analyzers.cmake)

option(ENABLE_PDLL_CONVERSIONS "Enable PDLL conversions" OFF)
if (ENABLE_PDLL_CONVERSIONS)
  message(STATUS "ENABLE_PDLL_CONVERSIONS")
  target_compile_definitions(vast_settings
    INTERFACE
      -DENABLE_PDLL_CONVERSIONS
  )
endif()

add_library(vast::settings ALIAS vast_settings)

#
# Add external libraries
#
add_subdirectory(external)

if (NOT ENABLE_GAP_SUBMODULE)
    find_package(gap CONFIG REQUIRED)
endif()

target_link_libraries(vast_settings
  INTERFACE
    gap::gap
    vast::util
)

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
    vast::translation_api
    vast::settings
)

target_include_directories(vast
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
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
    vast_settings
    vast_translation_api
    vast_codegen
    vast_frontend
    vast_frontend_tool
    vast_util

    MLIRCore
    MLIRMeta
    MLIRHighLevel

    VASTSymbolInterface
    VASTTypedAttrInterface
    VASTTypeQualifiersInterfaces
  )

  if (ENABLE_GAP_SUBMODULE)
    set(VAST_INSTALL_TARGETS
      ${VAST_INSTALL_TARGETS}

      gap
      gap-core
      gap-settings
    )
  endif()

  install(DIRECTORY ${CMAKE_BINARY_DIR}/include/vast
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT vast-headers
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.h.inc"
    PATTERN "*.hpp.inc"
    PATTERN "CMakeFiles" EXCLUDE
  )

  install(DIRECTORY include/vast
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT vast-headers
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "CMakeFiles" EXCLUDE
  )

  set(VAST_EXPORT_NAME VASTTargets)

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

  #
  # packaging support
  #

  set(CPACK_PACKAGE_VENDOR "Trail of Bits")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "VAST: an alternative frontend for the translation of Clang AST to MLIR."
  )
  set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR})
  set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR})
  set(CPACK_PACKAGE_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH})
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
  set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
  set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package")
  set(CPACK_PACKAGE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set(SYSTEM ${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_VERSION})
  set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PROJECT_VERSION}-${SYSTEM}")
  set(CPACK_SOURCE_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PROJECT_VERSION}-src")

  include(CPack)

  set(VAST_CONFIG_NAME "${PROJECT_NAME}Config")
  set(VAST_PACKAGE_CONFIG_FILE "${VAST_CONFIG_NAME}.cmake")
  set(VAST_PACKAGE_CONFIG_VERSION_FILE "${VAST_CONFIG_NAME}Version.cmake")

  include(CMakePackageConfigHelpers)

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${VAST_CONFIG_NAME}.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${VAST_PACKAGE_CONFIG_FILE}"
    INSTALL_DESTINATION ${VAST_CMAKE_INSTALL_DIR}
  )

  write_basic_package_version_file(
    ${VAST_PACKAGE_CONFIG_VERSION_FILE}
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${VAST_PACKAGE_CONFIG_FILE}"
    "${CMAKE_CURRENT_BINARY_DIR}/${VAST_PACKAGE_CONFIG_VERSION_FILE}"
    DESTINATION ${VAST_CMAKE_INSTALL_DIR}
  )

endif()
