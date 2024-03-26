// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenBlock.hpp"
#include "vast/CodeGen/Util.hpp"

namespace vast::cg
{
    operation block_generator::emit_in_scope(region_t &scope, const clang_compound_stmt *stmt) {
        auto _ = bld.insertion_guard();
        bld.set_insertion_point_to_end(&scope);
        return emit(stmt);
    }

    operation block_generator::emit(const clang_compound_stmt *stmt) {
        auto scope = bld.create< core::ScopeOp >(visitor.location(stmt));
        scope.getBody().emplaceBlock();

        auto _ = bld.insertion_guard();
        bld.set_insertion_point_to_end(&scope.getBody());

        for (auto &s : stmt->body()) {
            if (auto c = clang::dyn_cast< clang_compound_stmt >(s)) {
                auto &sg = mk_child< block_generator >(bld, visitor);
                sg.emit_in_scope(scope.getBody(), c);
            } else {
                visitor.visit(s);
            }
        }

        return scope;
    }

} // namespace vast::cg
