// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenScope.hpp"
#include "vast/Util/Warnings.hpp"

#include "vast/CodeGen/CodeGenDeclVisitor.hpp"
#include "vast/CodeGen/CodeGenStmtVisitor.hpp"
#include "vast/CodeGen/CodeGenTypeVisitor.hpp"
#include "vast/CodeGen/CodeGenAttrVisitor.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/FallBackVisitor.hpp"

namespace vast::cg
{
    struct default_visitor : visitor_base
    {
        operation visit(clang_decl *decl) override
        {
            VAST_UNIMPLEMENTED;
        }

        operation visit(clang_stmt *stmt) override
        {
            VAST_UNIMPLEMENTED;
        }

        mlir_type visit(clang_type *type) override
        {
            VAST_UNIMPLEMENTED;
        }

        mlir_type visit(clang_qual_type ty) override
        {
            VAST_UNIMPLEMENTED;
        }

        mlir_attr visit(clang_attr *attr) override
        {
            VAST_UNIMPLEMENTED;
        }

        mlir_type visit_as_lvalue_type(clang_qual_type ty) override
        {
            VAST_UNIMPLEMENTED;
        }

        mlir_type visit_function_type(const clang_function_type *fty, bool is_variadic) override
        {
            llvm::SmallVector< mlir_type > args;
            if (auto proto = clang::dyn_cast< clang_function_proto_type >(fty)) {
                for (auto param : proto->getParamTypes()) {
                    args.push_back(visit_as_lvalue_type(param));
                }
            }

            return core::FunctionType::get(args, {visit(fty->getReturnType())}, is_variadic);
        }
    };

} // namespace vast::cg
