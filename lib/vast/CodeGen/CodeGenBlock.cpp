// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenBlock.hpp"
#include "vast/CodeGen/Util.hpp"

namespace vast::cg
{
    void block_generator::emit_in_scope(region_t &scope, const clang_compound_stmt *stmt) {
        default_generator_base::emit_in_scope(scope, [&] { return emit(stmt); });
    }

    void block_generator::emit_in_new_scope(const clang_compound_stmt *stmt) {
        auto &sg = mk_child< block_generator >(bld, visitor);
        auto new_scope = make_scope(visitor.location(stmt));
        sg.emit_in_scope(new_scope.getBody(), stmt);
    }

    void block_generator::emit(const clang_decl_stmt *stmt) {
        if (stmt->isSingleDecl()) {
            emit(stmt->getSingleDecl());
        } else {
            // TODO make scoped?
            for (auto &d : stmt->decls()) {
                emit(d);
            }
        }
    }

    void block_generator::emit(const clang_decl *decl) {
        if (auto vd = clang::dyn_cast< clang_var_decl >(decl)) {
            mk_var(*this, vd);
        } else {
            visitor.visit(decl);
        }
    }

    core::ScopeOp block_generator::make_scope(loc_t loc) {
        auto scope = bld.create< core::ScopeOp >(loc);
        scope.getBody().emplaceBlock();
        return scope;
    }

    void block_generator::emit(const clang_compound_stmt *stmt) {
        for (auto &s : stmt->body()) {
            if (auto cs = clang::dyn_cast< clang_compound_stmt >(s)) {
                emit_in_new_scope(cs);
            } else if (auto d = clang::dyn_cast< clang_decl_stmt >(s)) {
                emit(d);
            } else {
                visitor.visit(s);
            }
        }
    }

} // namespace vast::cg
