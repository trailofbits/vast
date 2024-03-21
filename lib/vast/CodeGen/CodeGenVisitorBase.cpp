// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

namespace vast::cg
{
    operation visitor_view::visit(const clang_decl *decl) {
        return visitor.visit(decl);
    }

    operation visitor_view::visit(const clang_stmt *stmt) {
        return visitor.visit(stmt);
    }

    mlir_type visitor_view::visit(const clang_type *type) {
        return visitor.visit(type);
    }

    mlir_type visitor_view::visit(clang_qual_type ty) {
        return visitor.visit(ty);
    }

    mlir_attr visitor_view::visit(const clang_attr *attr) {
        return visitor.visit(attr);
    }

    operation visitor_view::visit_prototype(const clang_function *decl) {
        return visitor.visit_prototype(decl);
    }

    mlir_type visitor_view::visit(const clang_function_type *fty, bool is_variadic) {
        return visitor.visit(fty, is_variadic);
    }

    mlir_type visitor_view::visit_as_lvalue_type(clang_qual_type ty) {
        return visitor.visit_as_lvalue_type(ty);
    }

    mcontext_t& visitor_view::mcontext() {
        return visitor.mcontext();
    }

    const mcontext_t& visitor_view::mcontext() const {
        return visitor.mcontext();
    }

    mlir_type visitor_base::visit(const clang_function_type *fty, bool is_variadic)
    {
        llvm::SmallVector< mlir_type > args;
        if (auto proto = clang::dyn_cast< clang_function_proto_type >(fty)) {
            for (auto param : proto->getParamTypes()) {
                args.push_back(visit_as_lvalue_type(param));
            }
        }

        return core::FunctionType::get(args, {visit(fty->getReturnType())}, is_variadic);
    }

    mlir_type visitor_base::visit_as_lvalue_type(clang_qual_type ty)
    {
        auto element_type = visit(ty);
        if (mlir::isa< hl::LValueType >(element_type)) {
            return element_type;
        }
        return hl::LValueType::get(&mcontext(), element_type);
    }

} // namespace vast::cg
