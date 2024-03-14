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
        void Visit(clang::Decl *decl) override
        {
            VAST_UNIMPLEMENTED;
        }

        void Visit(clang::Stmt *stmt) override
        {
            VAST_UNIMPLEMENTED;
        }

        void Visit(clang::Type *type) override
        {
            VAST_UNIMPLEMENTED;
        }

        void Visit(clang::Attr *attr) override
        {
            VAST_UNIMPLEMENTED;
        }
    };

} // namespace vast::cg
