// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

namespace vast::cg {

    struct visitor_view
    {
        explicit visitor_view(visitor_base &visitor) : visitor(visitor) {}

        operation visit(const clang_decl *decl) { return visitor.visit(decl); }
        operation visit(const clang_stmt *stmt) { return visitor.visit(stmt); }
        mlir_type visit(const clang_type *type) { return visitor.visit(type); }
        mlir_type visit(clang_qual_type ty)     { return visitor.visit(ty); }
        mlir_attr visit(const clang_attr *attr) { return visitor.visit(attr); }

        mlir_type visit(const clang_function_type *fty, bool is_variadic) {
            return visitor.visit(fty, is_variadic);
        }

        mlir_type visit_as_lvalue_type(clang_qual_type ty) {
            return visitor.visit_as_lvalue_type(ty);
        }

        loc_t location(const auto *node) const { return visitor.location(node); }

        mcontext_t& mcontext() { return visitor.mcontext(); }
        const mcontext_t& mcontext() const { return visitor.mcontext(); }

        codegen_builder& builder() { return visitor.builder(); }

        template< typename op_t >
        auto compose() { return builder().template compose< op_t >(); }

        insertion_guard insertion_guard() { return visitor.insertion_guard(); }

        void set_insertion_point_to_start(region_ptr region) { visitor.set_insertion_point_to_start(region); }
        void set_insertion_point_to_end(region_ptr region)   { visitor.set_insertion_point_to_end(region); }

        void set_insertion_point_to_start(block_ptr block) { visitor.set_insertion_point_to_start(block); }
        void set_insertion_point_to_end(block_ptr block)   { visitor.set_insertion_point_to_end(block); }

        visitor_base *raw() { return &visitor; }

      protected:
        visitor_base &visitor;
    };

} // namespace vast::cg
