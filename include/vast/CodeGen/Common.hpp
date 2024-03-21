// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include "vast/Dialect/Core/CoreAttributes.hpp"
#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::cg {

    using clang_decl = clang::Decl;
    using clang_stmt = clang::Stmt;
    using clang_expr = clang::Expr;
    using clang_type = clang::Type;
    using clang_attr = clang::Attr;

    using linkage_kind = core::GlobalLinkageKind;
    using mlir_visibility = mlir::SymbolTable::Visibility;
    using mlir_attr_list = mlir::NamedAttrList;

    using clang_function = clang::FunctionDecl;
    using clang_function_type = clang::FunctionType;
    using clang_function_proto_type = clang::FunctionProtoType;

    using clang_named_decl = clang::NamedDecl;
    using clang_var_decl = clang::VarDecl;

    using clang_qual_type = clang::QualType;

    using vast_function = vast::hl::FuncOp;
    using vast_function_type = core::FunctionType;

} // namespace vast::cg
