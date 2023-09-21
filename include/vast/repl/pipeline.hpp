// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once


namespace vast::repl {

    struct pipeline {
        using pass_name = std::string;
        using passname_ref = std::string_view;

        std::vector< pass_name > passes;
        std::vector< passname_ref > make_snapshot_after;
    };

} // namespace vast::repl
