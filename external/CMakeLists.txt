#
# Copyright (c) 2023, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

option(ENABLE_GAP_SUBMODULE "Enable build with gap submodule" ON)

if (ENABLE_GAP_SUBMODULE)
    set(GAP_ENABLE_VCPKG OFF)
    set(GAP_ENABLE_TESTING OFF)
    set(GAP_ENABLE_EXAMPLES OFF)
    add_subdirectory(gap)
else()
    find_package(gap CONFIG REQUIRED)
endif()
