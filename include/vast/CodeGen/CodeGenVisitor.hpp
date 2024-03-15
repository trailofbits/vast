// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenScope.hpp"
#include "vast/Util/Warnings.hpp"

#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/FallBackVisitor.hpp"

namespace vast::cg
{
    struct default_visitor : fallback_visitor
    {
        using fallback_visitor::fallback_visitor;

        using fallback_visitor::visit;

        mlir_type visit(const clang_function_type *fty, bool is_variadic) override
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
