// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"
#include "vast/Util/TypeList.hpp"

namespace vast::cg
{
    //
    // fallback_visitor
    //
    // Allows to specify chain of fallback visitors in case that first `visitor::Visit` is
    // unsuccessful.
    //
    template< typename derived_t, template< typename > typename ...visitors >
    struct fallback_visitor : visitors< derived_t >...
    {
        operation Visit(const clang::Stmt *stmt) { return visit_with_fallback(stmt); }
        operation Visit(const clang::Decl *decl) { return visit_with_fallback(decl); }
        mlir_type Visit(const clang::Type *type) { return visit_with_fallback(type); }
        mlir_attr Visit(const clang::Attr *attr) { return visit_with_fallback(attr); }
        mlir_type Visit(clang::QualType    type) { return visit_with_fallback(type); }

        using visitors_list = util::type_list< visitors< derived_t >... >;

        auto visit_with_fallback(auto token) {
            using result_type = decltype(visitors_list::head::Visit(token));

            result_type result;
            ((result = visitors< derived_t >::Visit(token)) || ... );
            return result;
        }
    };

} // namespace vast::cg
