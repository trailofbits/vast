//
// Copyright (c) 2021-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in
// the LICENSE file found in the root directory of this source tree.
//

/* This generated file is for internal use. Do not include it from headers. */

#include <string_view>

#ifdef VAST_CONFIG_H
#error config.h can only be included once
#else
#define VAST_CONFIG_H

namespace vast {

    constexpr std::string_view version = "${VAST_VERSION}";

    constexpr std::string_view homepage_url = "${PROJECT_HOMEPAGE_URL}";

    constexpr std::string_view bug_report_url = "${BUG_REPORT_URL}";

    constexpr std::string_view default_resource_dir = "${VAST_DEFAULT_RESOURCE_DIR}";

    constexpr std::string_view default_sysroot = "${VAST_DEFAULT_SYSROOT}";

} // namespace vast

#endif
