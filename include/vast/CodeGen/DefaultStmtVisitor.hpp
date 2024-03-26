// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/OperationKinds.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

namespace vast::cg {

    struct default_stmt_visitor : stmt_visitor_base< default_stmt_visitor >
    {
        using base = stmt_visitor_base< default_stmt_visitor >;

        explicit default_stmt_visitor(codegen_builder &bld, visitor_view self)
            : base(bld, self)
        {}

        using base::Visit;

        operation visit(const clang_stmt *stmt) { return Visit(stmt); }

        //
        // ControlFlow Statements
        //

        operation VisitReturnStmt(const clang::ReturnStmt *stmt);

        //
        // Literals
        //
        operation VisistCharacterLiteral(const clang::CharacterLiteral *lit);
        operation VisitIntegerLiteral(const clang::IntegerLiteral *lit);
        operation VisitFloatingLiteral(const clang::FloatingLiteral *lit);
        operation VisitStringLiteral(const clang::StringLiteral *lit);
        operation VisitUserDefinedLiteral(const clang::UserDefinedLiteral *lit);
        operation VisitCompoundLiteralExpr(const clang::CompoundLiteralExpr *lit);
        operation VisitFixedPointLiteral(const clang::FixedPointLiteral *lit);
    };

} // namespace vast::cg
