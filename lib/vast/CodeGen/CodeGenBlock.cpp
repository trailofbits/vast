// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenBlock.hpp"
#include "vast/CodeGen/Util.hpp"

namespace vast::cg
{
    core::ScopeOp block_generator::make_block(loc_t loc) {
        auto block = bld.create< core::ScopeOp >(loc);
        block.getBody().emplaceBlock(); // FIXME: create block in ScopeOp ctor
        return block;
    }

    operation block_generator::emit(const clang_compound_stmt *stmt) {
        auto block = make_block(visitor.location(stmt).value());

        auto _ = bld.scoped_insertion_at_start(&block.getBody());
        for (auto &s : stmt->body()) {
            visitor.visit(s);
        }

        return block;
    }

} // namespace vast::cg
