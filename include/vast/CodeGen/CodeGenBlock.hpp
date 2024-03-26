// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/DefaultGeneratorBase.hpp"
#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/CodeGenOptions.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::cg {

    struct block_generator : default_generator_base, block_scope
    {
        block_generator(scope_context *parent, codegen_builder &bld, visitor_view visitor)
            : default_generator_base(bld, visitor), block_scope(parent)
        {}

        virtual ~block_generator() = default;

        operation emit_in_scope(region_t &scope, const clang_compound_stmt *stmt);
      private:

        operation emit(const clang_compound_stmt *stmt);
    };
} // namespace vast::cg
