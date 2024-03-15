// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenVisitor.hpp"

#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

namespace vast::cg
{
    mlir_type codegen_visitor::visit(const clang_function_type *fty, bool is_variadic)
    {
        llvm::SmallVector< mlir_type > args;
        if (auto proto = clang::dyn_cast< clang_function_proto_type >(fty)) {
            for (auto param : proto->getParamTypes()) {
                args.push_back(visit_as_lvalue_type(param));
            }
        }

        return core::FunctionType::get(args, {visit(fty->getReturnType())}, is_variadic);
    }

    mlir_type codegen_visitor::visit_as_lvalue_type(clang_qual_type ty)
    {
        auto element_type = visit(ty);
        if (mlir::isa< hl::LValueType >(element_type)) {
            return element_type;
        }
        return hl::LValueType::get(&mcontext(), element_type);
    }
} // namespace vast::cg
