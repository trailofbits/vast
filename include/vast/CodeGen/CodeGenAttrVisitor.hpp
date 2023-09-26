// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/AttrVisitor.h>
#include <clang/AST/Attr.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Frontend/FrontendDiagnostic.h>
VAST_UNRELAX_WARNINGS

namespace vast::cg {

    template< typename Derived >
    struct CodeGenAttrVisitor
        : clang::ConstAttrVisitor< CodeGenAttrVisitor< Derived >, operation >
        , CodeGenVisitorLens< CodeGenAttrVisitor< Derived >, Derived >
        , CodeGenBuilder< CodeGenAttrVisitor< Derived >, Derived >
    {
        using LensType = CodeGenVisitorLens< CodeGenAttrVisitor< Derived >, Derived >;

        using LensType::derived;
        using LensType::context;
        using LensType::mcontext;
        using LensType::acontext;

        using LensType::meta_location;

    };
} // namespace vast::cg
