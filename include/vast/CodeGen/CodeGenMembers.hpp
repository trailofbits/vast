// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/GeneratorBase.hpp"
#include "vast/CodeGen/ScopeContext.hpp"

namespace vast::cg {

    struct members_generator : generator_base
    {
        using scope_type = members_scope;

        using generator_base::generator_base;
        virtual ~members_generator() = default;

        void emit(const clang::RecordDecl *decl);
    };

} // namespace vast::cg
