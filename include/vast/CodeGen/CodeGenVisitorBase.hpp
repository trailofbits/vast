// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/AttrVisitor.h>
#include <clang/AST/DeclVisitor.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/TypeVisitor.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenContext.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/Mangler.hpp"

#include "vast/Util/Common.hpp"

namespace vast::cg {

    template< typename derived_t >
    using decl_visitor_base = clang::ConstDeclVisitor< derived_t, operation >;

    template< typename derived_t >
    using stmt_visitor_base = clang::ConstStmtVisitor< derived_t, operation >;

    template< typename derived_t >
    using type_visitor_base = clang::TypeVisitor< derived_t, mlir_type >;

    template< typename derived_t >
    using attr_visitor_base = clang::ConstAttrVisitor< derived_t, mlir_attr >;

    template< typename context_t, meta_generator_like meta_gen >
    struct visitor_base
    {
        visitor_base(context_t &ctx, meta_gen &meta)
            : ctx(ctx), meta(meta), _builder(ctx.getBodyRegion())
        {}

        void set_insertion_point_to_start(region_ptr region) {
            _builder.setInsertionPointToStart(&region->front());
        }

        void set_insertion_point_to_end(region_ptr region) {
            _builder.setInsertionPointToEnd(&region->back());
        }

        void set_insertion_point_to_start(block_ptr block) {
            _builder.setInsertionPointToStart(block);
        }

        void set_insertion_point_to_end(block_ptr block) {
            _builder.setInsertionPointToEnd(block);
        }

        void clear_insertion_point() {
            _builder.clearInsertionPoint();
        }

        insertion_guard make_insertion_guard() {
            return { _builder };
        }

        mlir_builder& base_builder() { return _builder; }

        template< typename Token >
        loc_t meta_location(Token token) const {
            return meta.get(token).location();
        }

        context_t &ctx;
        meta_gen &meta;

        mlir_builder _builder;
    };

} // namespace vast::cg
