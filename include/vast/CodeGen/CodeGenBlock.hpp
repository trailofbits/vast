// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/GeneratorBase.hpp"
#include "vast/CodeGen/ScopeContext.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::cg {

    struct block_generator : generator_base {
        using scope_type = block_scope;

        using generator_base::generator_base;
        virtual ~block_generator() = default;

        operation emit(const clang_compound_stmt *stmt);

        core::ScopeOp make_block(loc_t loc);
    };

} // namespace vast::cg
