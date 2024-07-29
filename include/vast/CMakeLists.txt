# Copyright (c) 2022-present, Trail of Bits, Inc.

if (VAST_BUILD_DIALECTS)
    add_subdirectory(Dialect)
    add_subdirectory(Interfaces)
endif()

if (VAST_BUILD_CONVERSIONS)
    add_subdirectory(Conversion)
endif()

add_subdirectory(Tower)
add_subdirectory(Util)
add_subdirectory(Analyses)
