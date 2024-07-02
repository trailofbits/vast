// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Tower/Tower.hpp"
#include "vast/repl/common.hpp"
#include "vast/repl/command_base.hpp"
#include "vast/repl/pipeline.hpp"

#include <filesystem>
#include <unordered_map>

namespace vast::repl {

    struct state_t {
        explicit state_t(mcontext_t &ctx) : ctx(ctx) {}

        //
        // perform exit in next step
        //
        bool exit = false;

        //
        // c/c++ source file to compile
        //
        std::optional< std::filesystem::path > source;

        //
        // mlir module and context
        //
        mcontext_t &ctx;

        //
        // Tower related state
        //
        tw::location_info li;
        std::optional< tw::default_tower > tower;

        std::unordered_map< std::string, tw::link_ptr > links;

        //
        // sticked commands performed after each step
        //
        std::vector< command_ptr > sticked;

        //
        // named pipelines
        //
        llvm::StringMap< pipeline > pipelines;

        //
        // verbosity flags
        //
        bool verbose_pipeline = true;

        void raise_tower(owning_module_ref mod);
        vast_module current_module();
    };

} // namespace vast::repl
