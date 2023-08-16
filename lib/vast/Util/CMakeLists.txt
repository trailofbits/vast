# Copyright (c) 2022-present, Trail of Bits, Inc.

add_library(vast_util
    Region.cpp
    Warnings.cpp
)

target_include_directories(vast_util
    PRIVATE
        $<BUILD_INTERFACE:${VAST_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${VAST_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)


target_link_libraries(vast_util PRIVATE vast::settings)

add_library(vast::util ALIAS vast_util)
