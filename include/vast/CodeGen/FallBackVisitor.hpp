// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"
#include "vast/Util/TypeList.hpp"

namespace vast::cg
{
    //
    // FallBackVisitor
    //
    // Allows to specify chain of fallback visitors in case that first `Visitor::Visit` is
    // unsuccessful.
    //
    template< typename Derived, template< typename > typename ...Visitors >
    struct FallBackVisitor : Visitors< Derived >...
    {
        operation Visit(const clang::Stmt *stmt) { return visit_with_fallback(stmt); }
        operation Visit(const clang::Decl *decl) { return visit_with_fallback(decl); }
        mlir_type Visit(const clang::Type *type) { return visit_with_fallback(type); }
        mlir_type Visit(clang::QualType    type) { return visit_with_fallback(type); }

        using visitors = util::type_list< Visitors< Derived >... >;

        auto visit_with_fallback(auto token) {
            using result_type = decltype(visitors::head::Visit(token));

            result_type result;
            ((result = Visitors< Derived >::Visit(token)) || ... );
            return result;
        }
    };

} // namespace vast::cg
