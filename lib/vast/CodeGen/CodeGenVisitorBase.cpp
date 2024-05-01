// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

namespace vast::cg
{
    operation visitor_view::visit(const clang_decl *decl, scope_context &scope) {
        return visitor.visit(decl, scope);
    }

    operation visitor_view::visit(const clang_stmt *stmt, scope_context &scope) {
        return visitor.visit(stmt, scope);
    }

    mlir_type visitor_view::visit(const clang_type *type, scope_context &scope) {
        return visitor.visit(type, scope);
    }

    mlir_type visitor_view::visit(clang_qual_type ty, scope_context &scope) {
        return visitor.visit(ty, scope);
    }

    mlir_attr visitor_view::visit(const clang_attr *attr, scope_context &scope) {
        return visitor.visit(attr, scope);
    }

    operation visitor_view::visit_prototype(const clang_function *decl, scope_context &scope) {
        return visitor.visit_prototype(decl, scope);
    }

    mlir_type visitor_view::visit(const clang_function_type *fty, bool is_variadic, scope_context &scope) {
        return visitor.visit(fty, is_variadic, scope);
    }

    mlir_type visitor_view::visit_as_lvalue_type(clang_qual_type ty, scope_context &scope) {
        return visitor.visit_as_lvalue_type(ty, scope);
    }

    mcontext_t& visitor_view::mcontext() {
        return visitor.mcontext();
    }

    const mcontext_t& visitor_view::mcontext() const {
        return visitor.mcontext();
    }

    operation scoped_visitor_view::visit(const clang_decl *decl) {
        return visitor_view::visit(decl, scope);
    }

    operation scoped_visitor_view::visit(const clang_stmt *stmt) {
        return visitor_view::visit(stmt, scope);
    }

    mlir_type scoped_visitor_view::visit(const clang_type *type) {
        return visitor_view::visit(type, scope);
    }

    mlir_type scoped_visitor_view::visit(clang_qual_type ty) {
        return visitor_view::visit(ty, scope);
    }

    mlir_attr scoped_visitor_view::visit(const clang_attr *attr) {
        return visitor_view::visit(attr, scope);
    }

    operation scoped_visitor_view::visit_prototype(const clang_function *decl) {
        return visitor_view::visit_prototype(decl, scope);
    }

    mlir_type scoped_visitor_view::visit(const clang_function_type *fty, bool is_variadic) {
        return visitor_view::visit(fty, is_variadic, scope);
    }

    mlir_type scoped_visitor_view::visit_as_lvalue_type(clang_qual_type ty) {
        return visitor_view::visit_as_lvalue_type(ty, scope);
    }

    mlir_type visitor_base::visit(const clang_function_type *fty, bool is_variadic, scope_context &scope)
    {
        llvm::SmallVector< mlir_type > args;
        if (auto proto = clang::dyn_cast< clang_function_proto_type >(fty)) {
            for (auto param : proto->getParamTypes()) {
                args.push_back(visit_as_lvalue_type(param, scope));
            }
        }

        return core::FunctionType::get(args, {visit(fty->getReturnType(), scope)}, is_variadic);
    }

    mlir_type visitor_base::visit_as_lvalue_type(clang_qual_type ty, scope_context &scope)
    {
        auto element_type = visit(ty, scope);
        if (mlir::isa< hl::LValueType >(element_type)) {
            return element_type;
        }
        return hl::LValueType::get(&mcontext(), element_type);
    }

} // namespace vast::cg
