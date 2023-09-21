// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include "gap/coro/generator.hpp"

#include <fstream>
#include <optional>

namespace vast::repl::ini {

    struct section_name {
        std::string name;

        static std::optional< section_name > parse(std::ifstream &in);
    };

    struct section {
        section_name name;
        std::vector< std::string > content;

        bool is_sticky_section() const;
        bool is_pipeline_section() const;

        string_ref last_name() const;

        void dump() const;

        static std::optional< section > parse(std::ifstream &in);
    };

    struct config {
        std::vector< section > sections;

        void dump() const;

        static config parse(std::ifstream &in);
    };


} // namespace vast::repl::ini
